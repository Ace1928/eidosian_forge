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
def _extract_operand(self, node, names, indices, temps):
    node = unwrap_node(node)
    if not node.type.is_pyobject:
        return False
    if isinstance(node, ExprNodes.CoerceToTempNode):
        temps.append(node)
        node = node.arg
    name_path = []
    obj_node = node
    while obj_node.is_attribute:
        if obj_node.is_py_attr:
            return False
        name_path.append(obj_node.member)
        obj_node = obj_node.obj
    if obj_node.is_name:
        name_path.append(obj_node.name)
        names.append(('.'.join(name_path[::-1]), node))
    elif node.is_subscript:
        if node.base.type != Builtin.list_type:
            return False
        if not node.index.type.is_int:
            return False
        if not node.base.is_name:
            return False
        indices.append(node)
    else:
        return False
    return True