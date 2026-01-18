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
def _handle_any_slot__new__(self, node, function, args, is_unbound_method, kwargs=None):
    """Replace 'exttype.__new__(exttype, ...)' by a call to exttype->tp_new()
        """
    obj = function.obj
    if not is_unbound_method or len(args) < 1:
        return node
    type_arg = args[0]
    if not obj.is_name or not type_arg.is_name:
        return node
    if obj.type != Builtin.type_type or type_arg.type != Builtin.type_type:
        return node
    if not type_arg.type_entry or not obj.type_entry:
        if obj.name != type_arg.name:
            return node
    elif type_arg.type_entry != obj.type_entry:
        return node
    args_tuple = ExprNodes.TupleNode(node.pos, args=args[1:])
    args_tuple = args_tuple.analyse_types(self.current_env(), skip_children=True)
    if type_arg.type_entry:
        ext_type = type_arg.type_entry.type
        if ext_type.is_extension_type and ext_type.typeobj_cname and (ext_type.scope.global_scope() == self.current_env().global_scope()):
            tp_slot = TypeSlots.ConstructorSlot('tp_new', '__new__')
            slot_func_cname = TypeSlots.get_slot_function(ext_type.scope, tp_slot)
            if slot_func_cname:
                cython_scope = self.context.cython_scope
                PyTypeObjectPtr = PyrexTypes.CPtrType(cython_scope.lookup('PyTypeObject').type)
                pyx_tp_new_kwargs_func_type = PyrexTypes.CFuncType(ext_type, [PyrexTypes.CFuncTypeArg('type', PyTypeObjectPtr, None), PyrexTypes.CFuncTypeArg('args', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('kwargs', PyrexTypes.py_object_type, None)])
                type_arg = ExprNodes.CastNode(type_arg, PyTypeObjectPtr)
                if not kwargs:
                    kwargs = ExprNodes.NullNode(node.pos, type=PyrexTypes.py_object_type)
                return ExprNodes.PythonCapiCallNode(node.pos, slot_func_cname, pyx_tp_new_kwargs_func_type, args=[type_arg, args_tuple, kwargs], may_return_none=False, is_temp=True)
    else:
        type_arg = type_arg.as_none_safe_node('object.__new__(X): X is not a type object (NoneType)')
    utility_code = UtilityCode.load_cached('tp_new', 'ObjectHandling.c')
    if kwargs:
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_tp_new_kwargs', self.Pyx_tp_new_kwargs_func_type, args=[type_arg, args_tuple, kwargs], utility_code=utility_code, is_temp=node.is_temp)
    else:
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_tp_new', self.Pyx_tp_new_func_type, args=[type_arg, args_tuple], utility_code=utility_code, is_temp=node.is_temp)