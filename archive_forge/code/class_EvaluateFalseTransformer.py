from tokenize import (generate_tokens, untokenize, TokenError,
from keyword import iskeyword
import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min
class EvaluateFalseTransformer(ast.NodeTransformer):
    operators = {ast.Add: 'Add', ast.Mult: 'Mul', ast.Pow: 'Pow', ast.Sub: 'Add', ast.Div: 'Mul', ast.BitOr: 'Or', ast.BitAnd: 'And', ast.BitXor: 'Not'}
    functions = ('Abs', 'im', 're', 'sign', 'arg', 'conjugate', 'acos', 'acot', 'acsc', 'asec', 'asin', 'atan', 'acosh', 'acoth', 'acsch', 'asech', 'asinh', 'atanh', 'cos', 'cot', 'csc', 'sec', 'sin', 'tan', 'cosh', 'coth', 'csch', 'sech', 'sinh', 'tanh', 'exp', 'ln', 'log', 'sqrt', 'cbrt')
    relational_operators = {ast.NotEq: 'Ne', ast.Lt: 'Lt', ast.LtE: 'Le', ast.Gt: 'Gt', ast.GtE: 'Ge', ast.Eq: 'Eq'}

    def visit_Compare(self, node):
        if node.ops[0].__class__ in self.relational_operators:
            sympy_class = self.relational_operators[node.ops[0].__class__]
            right = self.visit(node.comparators[0])
            left = self.visit(node.left)
            new_node = ast.Call(func=ast.Name(id=sympy_class, ctx=ast.Load()), args=[left, right], keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            return new_node
        return node

    def flatten(self, args, func):
        result = []
        for arg in args:
            if isinstance(arg, ast.Call):
                arg_func = arg.func
                if isinstance(arg_func, ast.Call):
                    arg_func = arg_func.func
                if arg_func.id == func:
                    result.extend(self.flatten(arg.args, func))
                else:
                    result.append(arg)
            else:
                result.append(arg)
        return result

    def visit_BinOp(self, node):
        if node.op.__class__ in self.operators:
            sympy_class = self.operators[node.op.__class__]
            right = self.visit(node.right)
            left = self.visit(node.left)
            rev = False
            if isinstance(node.op, ast.Sub):
                right = ast.Call(func=ast.Name(id='Mul', ctx=ast.Load()), args=[ast.UnaryOp(op=ast.USub(), operand=ast.Num(1)), right], keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            elif isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.UnaryOp):
                    left, right = (right, left)
                    rev = True
                    left = ast.Call(func=ast.Name(id='Pow', ctx=ast.Load()), args=[left, ast.UnaryOp(op=ast.USub(), operand=ast.Num(1))], keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
                else:
                    right = ast.Call(func=ast.Name(id='Pow', ctx=ast.Load()), args=[right, ast.UnaryOp(op=ast.USub(), operand=ast.Num(1))], keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            if rev:
                left, right = (right, left)
            new_node = ast.Call(func=ast.Name(id=sympy_class, ctx=ast.Load()), args=[left, right], keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            if sympy_class in ('Add', 'Mul'):
                new_node.args = self.flatten(new_node.args, sympy_class)
            return new_node
        return node

    def visit_Call(self, node):
        new_node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            new_node.keywords.append(ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load())))
        return new_node