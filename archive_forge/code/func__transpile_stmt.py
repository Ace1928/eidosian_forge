import ast
import collections
import inspect
import linecache
import numbers
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
import types
import numpy
from cupy_backends.cuda.api import runtime
from cupy._core._codeblock import CodeBlock, _CodeType
from cupy._core import _kernel
from cupy._core._dtype import _raise_if_invalid_cast
from cupyx import jit
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit import _builtin_funcs
from cupyx.jit import _interface
@transpile_function_wrapper
def _transpile_stmt(stmt: ast.stmt, is_toplevel: bool, env: Environment) -> _CodeType:
    """Transpile the statement.

    Returns (list of [CodeBlock or str]): The generated CUDA code.
    """
    if isinstance(stmt, ast.ClassDef):
        raise NotImplementedError('class is not supported currently.')
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise NotImplementedError('Nested functions are not supported currently.')
    if isinstance(stmt, ast.Return):
        value = _transpile_expr(stmt.value, env)
        value = Data.init(value, env)
        t = value.ctype
        if env.ret_type is None:
            env.ret_type = t
        elif env.ret_type != t:
            raise ValueError(f'Failed to infer the return type: {env.ret_type} or {t}')
        return [f'return {value.code};']
    if isinstance(stmt, ast.Delete):
        raise NotImplementedError('`del` is not supported currently.')
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            raise NotImplementedError('Not implemented.')
        value = _transpile_expr(stmt.value, env)
        var = stmt.targets[0]
        if is_constants(value) and isinstance(var, ast.Name):
            name = var.id
            if not isinstance(value.obj, _typeclasses):
                if is_toplevel:
                    if isinstance(env[name], Data):
                        raise TypeError(f'Type mismatch of variable: `{name}`')
                    env.consts[name] = value
                    return []
                else:
                    raise TypeError('Cannot assign constant value not at top-level.')
        value = Data.init(value, env)
        return _transpile_assign_stmt(var, env, value, is_toplevel)
    if isinstance(stmt, ast.AugAssign):
        value = _transpile_expr(stmt.value, env)
        target = _transpile_expr(stmt.target, env)
        if not isinstance(target, Data):
            raise TypeError(f'Cannot augassign to {target.code}')
        value = Data.init(value, env)
        tmp = Data(env.get_fresh_variable_name('_tmp_'), target.ctype)
        result = _eval_operand(stmt.op, (tmp, value), env)
        assert isinstance(target, Data)
        assert isinstance(result, Data)
        assert isinstance(target.ctype, _cuda_types.Scalar)
        assert isinstance(result.ctype, _cuda_types.Scalar)
        _raise_if_invalid_cast(result.ctype.dtype, target.ctype.dtype, 'same_kind')
        return ['{ ' + target.ctype.declvar('&' + tmp.code, target) + '; ' + target.ctype.assign(tmp, result) + '; }']
    if isinstance(stmt, ast.For):
        if len(stmt.orelse) > 0:
            raise NotImplementedError('while-else is not supported.')
        assert isinstance(stmt.target, ast.Name)
        name = stmt.target.id
        iters = _transpile_expr(stmt.iter, env)
        loop_var = env[name]
        if loop_var is None:
            target = Data(stmt.target.id, iters.ctype)
            env.locals[name] = target
            env.decls[name] = target
        elif isinstance(loop_var, Constant):
            raise TypeError('loop counter must not be constant value')
        elif loop_var.ctype != iters.ctype:
            raise TypeError(f'Data type mismatch of variable: `{name}`: {loop_var.ctype} != {iters.ctype}')
        if not isinstance(iters, _internal_types.Range):
            raise NotImplementedError('for-loop is supported only for range iterator.')
        body = _transpile_stmts(stmt.body, False, env)
        init_code = f'{iters.ctype} __it = {iters.start.code}, __stop = {iters.stop.code}, __step = {iters.step.code}'
        cond = '__step >= 0 ? __it < __stop : __it > __stop'
        if iters.step_is_positive is True:
            cond = '__it < __stop'
        elif iters.step_is_positive is False:
            cond = '__it > __stop'
        head = f'for ({init_code}; {cond}; __it += __step)'
        code: _CodeType = [CodeBlock(head, [f'{name} = __it;'] + body)]
        unroll = iters.unroll
        if unroll is True:
            code = ['#pragma unroll'] + code
        elif unroll is not None:
            code = [f'#pragma unroll({unroll})'] + code
        return code
    if isinstance(stmt, ast.AsyncFor):
        raise ValueError('`async for` is not allowed.')
    if isinstance(stmt, ast.While):
        if len(stmt.orelse) > 0:
            raise NotImplementedError('while-else is not supported.')
        condition = _transpile_expr(stmt.test, env)
        condition = _astype_scalar(condition, _cuda_types.bool_, 'unsafe', env)
        condition = Data.init(condition, env)
        body = _transpile_stmts(stmt.body, False, env)
        head = f'while ({condition.code})'
        return [CodeBlock(head, body)]
    if isinstance(stmt, ast.If):
        condition = _transpile_expr(stmt.test, env)
        if is_constants(condition):
            stmts = stmt.body if condition.obj else stmt.orelse
            return _transpile_stmts(stmts, is_toplevel, env)
        head = f'if ({condition.code})'
        then_body = _transpile_stmts(stmt.body, False, env)
        else_body = _transpile_stmts(stmt.orelse, False, env)
        return [CodeBlock(head, then_body), CodeBlock('else', else_body)]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        raise ValueError('Switching contexts are not allowed.')
    if isinstance(stmt, (ast.Raise, ast.Try)):
        raise ValueError('throw/catch are not allowed.')
    if isinstance(stmt, ast.Assert):
        value = _transpile_expr(stmt.test, env)
        if is_constants(value):
            assert value.obj
            return [';']
        else:
            return ['assert(' + value + ');']
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        raise ValueError('Cannot import modules from the target functions.')
    if isinstance(stmt, (ast.Global, ast.Nonlocal)):
        raise ValueError('Cannot use global/nonlocal in the target functions.')
    if isinstance(stmt, ast.Expr):
        value = _transpile_expr(stmt.value, env)
        return [';'] if is_constants(value) else [value.code + ';']
    if isinstance(stmt, ast.Pass):
        return [';']
    if isinstance(stmt, ast.Break):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.Continue):
        raise NotImplementedError('Not implemented.')
    assert False