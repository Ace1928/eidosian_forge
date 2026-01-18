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
def _transform_bytes_iteration(self, node, slice_node, reversed=False):
    target_type = node.target.type
    if not target_type.is_int and target_type is not Builtin.bytes_type:
        return node
    unpack_temp_node = UtilNodes.LetRefNode(slice_node.as_none_safe_node("'NoneType' is not iterable"))
    slice_base_node = ExprNodes.PythonCapiCallNode(slice_node.pos, 'PyBytes_AS_STRING', self.PyBytes_AS_STRING_func_type, args=[unpack_temp_node], is_temp=0)
    len_node = ExprNodes.PythonCapiCallNode(slice_node.pos, 'PyBytes_GET_SIZE', self.PyBytes_GET_SIZE_func_type, args=[unpack_temp_node], is_temp=0)
    return UtilNodes.LetNode(unpack_temp_node, self._transform_carray_iteration(node, ExprNodes.SliceIndexNode(slice_node.pos, base=slice_base_node, start=None, step=None, stop=len_node, type=slice_base_node.type, is_temp=1), reversed=reversed))