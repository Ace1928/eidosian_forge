from pythran.analyses import ConstantExpressions, ASTMatcher
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import to_ast, ConversionError, ToNotEval, mangle
from pythran.analyses.ast_matcher import DamnTooLongPattern
from pythran.syntax import PythranSyntaxError
from pythran.utils import isintegral, isnum
from pythran.config import cfg
import builtins
import gast as ast
from copy import deepcopy
import logging
import sys
class ConstEval(ast.NodeVisitor):

    def __init__(self, globals):
        self.locals = {}
        self.globals = globals
        self.counter = 0
        self.counter_max = cfg.getint('pythran', 'fold_max_steps')

    def visit(self, node):
        self.counter += 1
        if self.counter == self.counter_max:
            raise ToNotEval()
        return getattr(self, 'visit_' + type(node).__name__)(node)

    def visit_Return(self, node):
        self.locals['@'] = node.value and self.visit(node.value)

    def visit_Delete(self, node):
        if isinstance(node, ast.Name):
            self.locals.pop(node.id)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.locals[target.id] = value
            elif isinstance(target, ast.Subscript):
                self.visit(target.value)[self.visit(target.slice)] = value
            else:
                raise NotImplementedError('assign')

    def visit_AugAssign(self, node):
        value = self.visit(node.value)
        ty = type(node.op)
        if isinstance(node.target, ast.Name):
            if ty is ast.Add:
                self.locals[node.target.id] += value
            elif ty is ast.Sub:
                self.locals[node.target.id] -= value
            elif ty is ast.Mult:
                self.locals[node.target.id] *= value
            elif ty is ast.MatMult:
                self.locals[node.target.id] @= value
            elif ty is ast.Div:
                self.locals[node.target.id] /= value
            elif ty is ast.Mod:
                self.locals[node.target.id] %= value
            elif ty is ast.Pow:
                self.locals[node.target.id] **= value
            elif ty is ast.LShift:
                self.locals[node.target.id] <<= value
            elif ty is ast.RShift:
                self.locals[node.target.id] >>= value
            elif ty is ast.BitOr:
                self.locals[node.target.id] |= value
            elif ty is ast.BitXor:
                self.locals[node.target.id] ^= value
            elif ty is ast.BitAnd:
                self.locals[node.target.id] &= value
            elif ty is ast.FloorDiv:
                self.locals[node.target.id] //= value
            else:
                raise ValueError('invalid binary op')
        elif isinstance(node.target, ast.Subscript):
            subscript = self.visit(node.target.subscript)
            if ty is ast.Add:
                self.visit(node.target.value)[subscript] += value
            elif ty is ast.Sub:
                self.visit(node.target.value)[subscript] -= value
            elif ty is ast.Mult:
                self.visit(node.target.value)[subscript] *= value
            elif ty is ast.MatMult:
                self.visit(node.target.value)[subscript] @= value
            elif ty is ast.Div:
                self.visit(node.target.value)[subscript] /= value
            elif ty is ast.Mod:
                self.visit(node.target.value)[subscript] %= value
            elif ty is ast.Pow:
                self.visit(node.target.value)[subscript] **= value
            elif ty is ast.LShift:
                self.visit(node.target.value)[subscript] <<= value
            elif ty is ast.RShift:
                self.visit(node.target.value)[subscript] >>= value
            elif ty is ast.BitOr:
                self.visit(node.target.value)[subscript] |= value
            elif ty is ast.BitXor:
                self.visit(node.target.value)[subscript] ^= value
            elif ty is ast.BitAnd:
                self.visit(node.target.value)[subscript] &= value
            elif ty is ast.FloorDiv:
                self.visit(node.target.value)[subscript] //= value
            else:
                raise ValueError('invalid binary op')
        else:
            raise NotImplementedError('assign')

    def visit_For(self, node):
        iter = self.visit(node.iter)
        for elt in iter:
            if isinstance(node.target, ast.Name):
                self.locals[node.target.id] = elt
            else:
                raise ValueError('invalid loop target')
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except ContinueLoop:
                continue
            except BreakLoop:
                break
        else:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_While(self, node):
        raise ToNotEval
        while self.visit(node.test):
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except ContinueLoop:
                continue
            except BreakLoop:
                break
        else:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_If(self, node):
        if self.visit(node.test):
            for stmt in node.body:
                self.visit(stmt)
        else:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_Break(self, node):
        raise BreakLoop

    def visit_Continue(self, node):
        raise ContinueLoop

    def visit_Pass(self, node):
        pass

    def visit_Yield(self, node):
        raise ToNotEval

    def visit_BoolOp(self, node):
        values = (self.visit(value) for value in node.values)
        if type(node.op) is ast.And:
            for value in values:
                if not value:
                    return value
            return value
        elif type(node.op) is ast.Or:
            for value in values:
                if value:
                    return value
            return value
        else:
            raise ValueError('invalid bool op')

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if type(node.op) is ast.Add:
            return left + right
        elif type(node.op) is ast.Sub:
            return left - right
        elif type(node.op) is ast.Mult:
            return left * right
        elif type(node.op) is ast.MatMult:
            return left @ right
        elif type(node.op) is ast.Div:
            return left / right
        elif type(node.op) is ast.Mod:
            return left % right
        elif type(node.op) is ast.Pow:
            return left ** right
        elif type(node.op) is ast.LShift:
            return left << right
        elif type(node.op) is ast.RShift:
            return left >> right
        elif type(node.op) is ast.BitOr:
            return left | right
        elif type(node.op) is ast.BitXor:
            return left ^ right
        elif type(node.op) is ast.BitAnd:
            return left & right
        elif type(node.op) is ast.FloorDiv:
            return left // right
        else:
            raise ValueError('invalid binary op')

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)
        if type(node.op) is ast.Invert:
            return ~value
        elif type(node.op) is ast.Not:
            return not value
        elif type(node.op) is ast.UAdd:
            return +value
        elif type(node.op) is ast.USub:
            return -value
        else:
            raise ValueError('invalid unary op')

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        if test:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Dict(self, node):
        return {self.visit(key): self.visit(value) for key, value in zip(node.keys, node.values)}

    def visit_Set(self, node):
        return {self.visit(elt) for elt in node.elts}

    def visit_Compare(self, node):
        curr = left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if type(op) is ast.Eq:
                cond = curr == right
            elif type(op) is ast.NotEq:
                cond = curr != right
            elif type(op) is ast.Lt:
                cond = curr < right
            elif type(op) is ast.LtE:
                cond = curr <= right
            elif type(op) is ast.Gt:
                cond = curr <= right
            elif type(op) is ast.GtE:
                cond = curr <= right
            elif type(op) is ast.Is:
                cond = curr <= right
            elif type(op) is ast.IsNot:
                cond = curr <= right
            elif type(op) is ast.In:
                cond = curr <= right
            else:
                raise ValueError('invalid compare op')
            if not cond:
                return False
            curr = right
        return True

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_Constant(self, node):
        return node.value

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        return getattr(value, node.attr)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return value[slice]

    def visit_Name(self, node):
        if node.id in self.locals:
            return self.locals[node.id]
        return self.globals[node.id]

    def visit_List(self, node):
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node):
        return tuple((self.visit(elt) for elt in node.elts))

    def visit_Slice(self, node):
        lower = node.lower and self.visit(node.lower)
        upper = node.upper and self.visit(node.upper)
        step = node.step and self.visit(node.step)
        return slice(lower, upper, step)

    def visit_ExtSlice(self, node):
        return tuple((self.visit(dim) for dim in node.dims))

    def visit_Index(self, node):
        return self.visit(node.value)

    def __call__(self, node):
        self.counter = 0
        try:
            return self.visit(node)
        except:
            self.locals.clear()
            raise
        finally:
            assert not self.locals