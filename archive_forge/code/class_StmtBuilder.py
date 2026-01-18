import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
class StmtBuilder(Builder):
    augassign_map = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Mod: '%', ast.BitOr: '|', ast.BitAnd: '&', ast.BitXor: '^', ast.LShift: '<<', ast.RShift: '>>', ast.Pow: '**'}

    @staticmethod
    def build_Expr(ctx, stmt):
        value = stmt.value
        if value.__class__.__name__ == 'Str':
            return None
        else:
            return ExprStmt(build_expr(ctx, value))

    @staticmethod
    def build_Assign(ctx, stmt):
        rhs = build_expr(ctx, stmt.value)
        lhs = [build_expr(ctx, x) for x in stmt.targets]
        return Assign(lhs, rhs)

    @staticmethod
    def build_AnnAssign(ctx, stmt):
        if stmt.value is None:
            raise UnsupportedNodeError(ctx, stmt, reason='without assigned value')
        if type(stmt.target) == ast.Attribute and stmt.target.value.id == 'self' and (ctx.funcname != '__init__'):
            start = stmt.col_offset
            end = start + len(f'self.{stmt.target.attr}')
            if hasattr(stmt.annotation, 'id'):
                end += len(f': {stmt.annotation.id}')
            sr = ctx.make_range(stmt.lineno, start, end)
            raise ValueError(f"Type annotations on instance attributes must be declared in __init__, not '{ctx.funcname}': {sr}")
        rhs = build_expr(ctx, stmt.value)
        lhs = build_expr(ctx, stmt.target)
        the_type = build_expr(ctx, stmt.annotation)
        return Assign([lhs], rhs, the_type)

    @staticmethod
    def build_Delete(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('del'))
        return Delete(r, [build_expr(ctx, target) for target in stmt.targets])

    @staticmethod
    def build_Return(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('return'))
        return Return(r, None if stmt.value is None else build_expr(ctx, stmt.value))

    @staticmethod
    def build_Raise(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('raise'))
        expr = build_expr(ctx, stmt.exc)
        return Raise(r, expr)

    @staticmethod
    def build_Assert(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('assert'))
        test = build_expr(ctx, stmt.test)
        msg = build_expr(ctx, stmt.msg) if stmt.msg is not None else None
        return Assert(r, test, msg)

    @staticmethod
    def build_AugAssign(ctx, stmt):
        lhs = build_expr(ctx, stmt.target)
        rhs = build_expr(ctx, stmt.value)
        op = type(stmt.op)
        if op in StmtBuilder.augassign_map:
            op_token = StmtBuilder.augassign_map[op]
        else:
            raise NotSupportedError(find_before(ctx, rhs.range().start, '=', offsets=(-1, 0)), 'unsupported kind of augmented assignment: ' + op.__name__)
        return AugAssign(lhs, op_token, rhs)

    @staticmethod
    def build_While(ctx, stmt):
        if stmt.orelse:
            raise NotSupportedError(None, "else branches of while loops aren't supported")
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('while'))
        return While(r, build_expr(ctx, stmt.test), build_stmts(ctx, stmt.body))

    @staticmethod
    def build_For(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('for'))
        if stmt.orelse:
            raise NotSupportedError(r, "else branches of for loops aren't supported")
        return For(r, [build_expr(ctx, stmt.target)], [build_expr(ctx, stmt.iter)], build_stmts(ctx, stmt.body))

    @staticmethod
    def build_If(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('if'))
        return If(r, build_expr(ctx, stmt.test), build_stmts(ctx, stmt.body), build_stmts(ctx, stmt.orelse))

    @staticmethod
    def build_Print(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('print'))
        if stmt.dest:
            raise NotSupportedError(r, "print statements with non-default destinations aren't supported")
        args = [build_expr(ctx, val) for val in stmt.values]
        return ExprStmt(Apply(Var(Ident(r, 'print')), args, []))

    @staticmethod
    def build_Pass(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('pass'))
        return Pass(r)

    @staticmethod
    def build_Break(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('break'))
        return Break(r)

    @staticmethod
    def build_Continue(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('continue'))
        return Continue(r)

    @staticmethod
    def build_With(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('with'))
        if is_torch_jit_ignore_context_manager(stmt):
            if not _IS_ASTUNPARSE_INSTALLED:
                raise RuntimeError('torch.jit._IgnoreContextManager requires installing Python library `astunparse`,                                   please install it in your Python environment')
            assign_ast = build_ignore_context_manager(ctx, stmt)
            return build_stmt(ctx, assign_ast)
        return With(r, build_withitems(ctx, stmt.items), build_stmts(ctx, stmt.body))