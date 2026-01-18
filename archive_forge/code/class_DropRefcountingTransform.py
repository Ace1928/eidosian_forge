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
class DropRefcountingTransform(Visitor.VisitorTransform):
    """Drop ref-counting in safe places.
    """
    visit_Node = Visitor.VisitorTransform.recurse_to_children

    def visit_ParallelAssignmentNode(self, node):
        """
        Parallel swap assignments like 'a,b = b,a' are safe.
        """
        left_names, right_names = ([], [])
        left_indices, right_indices = ([], [])
        temps = []
        for stat in node.stats:
            if isinstance(stat, Nodes.SingleAssignmentNode):
                if not self._extract_operand(stat.lhs, left_names, left_indices, temps):
                    return node
                if not self._extract_operand(stat.rhs, right_names, right_indices, temps):
                    return node
            elif isinstance(stat, Nodes.CascadedAssignmentNode):
                return node
            else:
                return node
        if left_names or right_names:
            lnames = [path for path, n in left_names]
            rnames = [path for path, n in right_names]
            if set(lnames) != set(rnames):
                return node
            if len(set(lnames)) != len(right_names):
                return node
        if left_indices or right_indices:
            lindices = []
            for lhs_node in left_indices:
                index_id = self._extract_index_id(lhs_node)
                if not index_id:
                    return node
                lindices.append(index_id)
            rindices = []
            for rhs_node in right_indices:
                index_id = self._extract_index_id(rhs_node)
                if not index_id:
                    return node
                rindices.append(index_id)
            if set(lindices) != set(rindices):
                return node
            if len(set(lindices)) != len(right_indices):
                return node
            return node
        temp_args = [t.arg for t in temps]
        for temp in temps:
            temp.use_managed_ref = False
        for _, name_node in left_names + right_names:
            if name_node not in temp_args:
                name_node.use_managed_ref = False
        for index_node in left_indices + right_indices:
            index_node.use_managed_ref = False
        return node

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

    def _extract_index_id(self, index_node):
        base = index_node.base
        index = index_node.index
        if isinstance(index, ExprNodes.NameNode):
            index_val = index.name
        elif isinstance(index, ExprNodes.ConstNode):
            return None
        else:
            return None
        return (base.name, index_val)