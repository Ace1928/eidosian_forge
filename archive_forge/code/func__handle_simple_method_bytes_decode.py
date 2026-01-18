from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def _handle_simple_method_bytes_decode(self, node, function, args, is_unbound_method):
    """Replace char*.decode() by a direct C-API call to the
        corresponding codec, possibly resolving a slice on the char*.
        """
    if not 1 <= len(args) <= 3:
        self._error_wrong_arg_count('bytes.decode', node, args, '1-3')
        return node
    string_node = args[0]
    start = stop = None
    if isinstance(string_node, ExprNodes.SliceIndexNode):
        index_node = string_node
        string_node = index_node.base
        start, stop = (index_node.start, index_node.stop)
        if not start or start.constant_result == 0:
            start = None
    if isinstance(string_node, ExprNodes.CoerceToPyTypeNode):
        string_node = string_node.arg
    string_type = string_node.type
    if string_type in (Builtin.bytes_type, Builtin.bytearray_type):
        if is_unbound_method:
            string_node = string_node.as_none_safe_node("descriptor '%s' requires a '%s' object but received a 'NoneType'", format_args=['decode', string_type.name])
        else:
            string_node = string_node.as_none_safe_node("'NoneType' object has no attribute '%.30s'", error='PyExc_AttributeError', format_args=['decode'])
    elif not string_type.is_string and (not string_type.is_cpp_string):
        return node
    parameters = self._unpack_encoding_and_error_mode(node.pos, args)
    if parameters is None:
        return node
    encoding, encoding_node, error_handling, error_handling_node = parameters
    if not start:
        start = ExprNodes.IntNode(node.pos, value='0', constant_result=0)
    elif not start.type.is_int:
        start = start.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
    if stop and (not stop.type.is_int):
        stop = stop.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
    codec_name = None
    if encoding is not None:
        codec_name = self._find_special_codec_name(encoding)
    if codec_name is not None:
        if codec_name in ('UTF16', 'UTF-16LE', 'UTF-16BE'):
            codec_cname = '__Pyx_PyUnicode_Decode%s' % codec_name.replace('-', '')
        else:
            codec_cname = 'PyUnicode_Decode%s' % codec_name
        decode_function = ExprNodes.RawCNameExprNode(node.pos, type=self.PyUnicode_DecodeXyz_func_ptr_type, cname=codec_cname)
        encoding_node = ExprNodes.NullNode(node.pos)
    else:
        decode_function = ExprNodes.NullNode(node.pos)
    temps = []
    if string_type.is_string:
        if not stop:
            if not string_node.is_name:
                string_node = UtilNodes.LetRefNode(string_node)
                temps.append(string_node)
            stop = ExprNodes.PythonCapiCallNode(string_node.pos, '__Pyx_ssize_strlen', self.Pyx_ssize_strlen_func_type, args=[string_node], is_temp=True)
        helper_func_type = self._decode_c_string_func_type
        utility_code_name = 'decode_c_string'
    elif string_type.is_cpp_string:
        if not stop:
            stop = ExprNodes.IntNode(node.pos, value='PY_SSIZE_T_MAX', constant_result=ExprNodes.not_a_constant)
        if self._decode_cpp_string_func_type is None:
            self._decode_cpp_string_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('string', string_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('stop', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('encoding', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('errors', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('decode_func', self.PyUnicode_DecodeXyz_func_ptr_type, None)])
        helper_func_type = self._decode_cpp_string_func_type
        utility_code_name = 'decode_cpp_string'
    else:
        if not stop:
            stop = ExprNodes.IntNode(node.pos, value='PY_SSIZE_T_MAX', constant_result=ExprNodes.not_a_constant)
        helper_func_type = self._decode_bytes_func_type
        if string_type is Builtin.bytes_type:
            utility_code_name = 'decode_bytes'
        else:
            utility_code_name = 'decode_bytearray'
    node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_%s' % utility_code_name, helper_func_type, args=[string_node, start, stop, encoding_node, error_handling_node, decode_function], is_temp=node.is_temp, utility_code=UtilityCode.load_cached(utility_code_name, 'StringTools.c'))
    for temp in temps[::-1]:
        node = UtilNodes.EvalWithTempExprNode(temp, node)
    return node