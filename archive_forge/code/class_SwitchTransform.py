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
class SwitchTransform(Visitor.EnvTransform):
    """
    This transformation tries to turn long if statements into C switch statements.
    The requirement is that every clause be an (or of) var == value, where the var
    is common among all clauses and both var and value are ints.
    """
    NO_MATCH = (None, None, None)

    def extract_conditions(self, cond, allow_not_in):
        while True:
            if isinstance(cond, (ExprNodes.CoerceToTempNode, ExprNodes.CoerceToBooleanNode)):
                cond = cond.arg
            elif isinstance(cond, ExprNodes.BoolBinopResultNode):
                cond = cond.arg.arg
            elif isinstance(cond, UtilNodes.EvalWithTempExprNode):
                cond = cond.subexpression
            elif isinstance(cond, ExprNodes.TypecastNode):
                cond = cond.operand
            else:
                break
        if isinstance(cond, ExprNodes.PrimaryCmpNode):
            if cond.cascade is not None:
                return self.NO_MATCH
            elif cond.is_c_string_contains() and isinstance(cond.operand2, (ExprNodes.UnicodeNode, ExprNodes.BytesNode)):
                not_in = cond.operator == 'not_in'
                if not_in and (not allow_not_in):
                    return self.NO_MATCH
                if isinstance(cond.operand2, ExprNodes.UnicodeNode) and cond.operand2.contains_surrogates():
                    return self.NO_MATCH
                return (not_in, cond.operand1, self.extract_in_string_conditions(cond.operand2))
            elif not cond.is_python_comparison():
                if cond.operator == '==':
                    not_in = False
                elif allow_not_in and cond.operator == '!=':
                    not_in = True
                else:
                    return self.NO_MATCH
                if is_common_value(cond.operand1, cond.operand1):
                    if cond.operand2.is_literal:
                        return (not_in, cond.operand1, [cond.operand2])
                    elif getattr(cond.operand2, 'entry', None) and cond.operand2.entry.is_const:
                        return (not_in, cond.operand1, [cond.operand2])
                if is_common_value(cond.operand2, cond.operand2):
                    if cond.operand1.is_literal:
                        return (not_in, cond.operand2, [cond.operand1])
                    elif getattr(cond.operand1, 'entry', None) and cond.operand1.entry.is_const:
                        return (not_in, cond.operand2, [cond.operand1])
        elif isinstance(cond, ExprNodes.BoolBinopNode):
            if cond.operator == 'or' or (allow_not_in and cond.operator == 'and'):
                allow_not_in = cond.operator == 'and'
                not_in_1, t1, c1 = self.extract_conditions(cond.operand1, allow_not_in)
                not_in_2, t2, c2 = self.extract_conditions(cond.operand2, allow_not_in)
                if t1 is not None and not_in_1 == not_in_2 and is_common_value(t1, t2):
                    if not not_in_1 or allow_not_in:
                        return (not_in_1, t1, c1 + c2)
        return self.NO_MATCH

    def extract_in_string_conditions(self, string_literal):
        if isinstance(string_literal, ExprNodes.UnicodeNode):
            charvals = list(map(ord, set(string_literal.value)))
            charvals.sort()
            return [ExprNodes.IntNode(string_literal.pos, value=str(charval), constant_result=charval) for charval in charvals]
        else:
            characters = string_literal.value
            characters = list({characters[i:i + 1] for i in range(len(characters))})
            characters.sort()
            return [ExprNodes.CharNode(string_literal.pos, value=charval, constant_result=charval) for charval in characters]

    def extract_common_conditions(self, common_var, condition, allow_not_in):
        not_in, var, conditions = self.extract_conditions(condition, allow_not_in)
        if var is None:
            return self.NO_MATCH
        elif common_var is not None and (not is_common_value(var, common_var)):
            return self.NO_MATCH
        elif not (var.type.is_int or var.type.is_enum) or any([not (cond.type.is_int or cond.type.is_enum) for cond in conditions]):
            return self.NO_MATCH
        return (not_in, var, conditions)

    def has_duplicate_values(self, condition_values):
        seen = set()
        for value in condition_values:
            if value.has_constant_result():
                if value.constant_result in seen:
                    return True
                seen.add(value.constant_result)
            else:
                try:
                    value_entry = value.entry
                    if (value_entry.type.is_enum or value_entry.type.is_cpp_enum) and value_entry.enum_int_value is not None:
                        value_for_seen = value_entry.enum_int_value
                    else:
                        value_for_seen = value_entry.cname
                except AttributeError:
                    return True
                if value_for_seen in seen:
                    return True
                seen.add(value_for_seen)
        return False

    def visit_IfStatNode(self, node):
        if not self.current_directives.get('optimize.use_switch'):
            self.visitchildren(node)
            return node
        common_var = None
        cases = []
        for if_clause in node.if_clauses:
            _, common_var, conditions = self.extract_common_conditions(common_var, if_clause.condition, False)
            if common_var is None:
                self.visitchildren(node)
                return node
            cases.append(Nodes.SwitchCaseNode(pos=if_clause.pos, conditions=conditions, body=if_clause.body))
        condition_values = [cond for case in cases for cond in case.conditions]
        if len(condition_values) < 2:
            self.visitchildren(node)
            return node
        if self.has_duplicate_values(condition_values):
            self.visitchildren(node)
            return node
        self.visitchildren(node, 'else_clause')
        for case in cases:
            self.visitchildren(case, 'body')
        common_var = unwrap_node(common_var)
        switch_node = Nodes.SwitchStatNode(pos=node.pos, test=common_var, cases=cases, else_clause=node.else_clause)
        return switch_node

    def visit_CondExprNode(self, node):
        if not self.current_directives.get('optimize.use_switch'):
            self.visitchildren(node)
            return node
        not_in, common_var, conditions = self.extract_common_conditions(None, node.test, True)
        if common_var is None or len(conditions) < 2 or self.has_duplicate_values(conditions):
            self.visitchildren(node)
            return node
        return self.build_simple_switch_statement(node, common_var, conditions, not_in, node.true_val, node.false_val)

    def visit_BoolBinopNode(self, node):
        if not self.current_directives.get('optimize.use_switch'):
            self.visitchildren(node)
            return node
        not_in, common_var, conditions = self.extract_common_conditions(None, node, True)
        if common_var is None or len(conditions) < 2 or self.has_duplicate_values(conditions):
            self.visitchildren(node)
            node.wrap_operands(self.current_env())
            return node
        return self.build_simple_switch_statement(node, common_var, conditions, not_in, ExprNodes.BoolNode(node.pos, value=True, constant_result=True), ExprNodes.BoolNode(node.pos, value=False, constant_result=False))

    def visit_PrimaryCmpNode(self, node):
        if not self.current_directives.get('optimize.use_switch'):
            self.visitchildren(node)
            return node
        not_in, common_var, conditions = self.extract_common_conditions(None, node, True)
        if common_var is None or len(conditions) < 2 or self.has_duplicate_values(conditions):
            self.visitchildren(node)
            return node
        return self.build_simple_switch_statement(node, common_var, conditions, not_in, ExprNodes.BoolNode(node.pos, value=True, constant_result=True), ExprNodes.BoolNode(node.pos, value=False, constant_result=False))

    def build_simple_switch_statement(self, node, common_var, conditions, not_in, true_val, false_val):
        result_ref = UtilNodes.ResultRefNode(node)
        true_body = Nodes.SingleAssignmentNode(node.pos, lhs=result_ref, rhs=true_val.coerce_to(node.type, self.current_env()), first=True)
        false_body = Nodes.SingleAssignmentNode(node.pos, lhs=result_ref, rhs=false_val.coerce_to(node.type, self.current_env()), first=True)
        if not_in:
            true_body, false_body = (false_body, true_body)
        cases = [Nodes.SwitchCaseNode(pos=node.pos, conditions=conditions, body=true_body)]
        common_var = unwrap_node(common_var)
        switch_node = Nodes.SwitchStatNode(pos=node.pos, test=common_var, cases=cases, else_clause=false_body)
        replacement = UtilNodes.TempResultFromStatNode(result_ref, switch_node)
        return replacement

    def visit_EvalWithTempExprNode(self, node):
        if not self.current_directives.get('optimize.use_switch'):
            self.visitchildren(node)
            return node
        orig_expr = node.subexpression
        temp_ref = node.lazy_temp
        self.visitchildren(node)
        if node.subexpression is not orig_expr:
            if not Visitor.tree_contains(node.subexpression, temp_ref):
                return node.subexpression
        return node
    visit_Node = Visitor.VisitorTransform.recurse_to_children