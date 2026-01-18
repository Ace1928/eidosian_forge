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
def _transpile_expr_internal(expr: ast.expr, env: Environment) -> _internal_types.Expr:
    if isinstance(expr, ast.BoolOp):
        values = [_transpile_expr(e, env) for e in expr.values]
        value = values[0]
        for rhs in values[1:]:
            value = _eval_operand(expr.op, (value, rhs), env)
        return value
    if isinstance(expr, ast.BinOp):
        left = _transpile_expr(expr.left, env)
        right = _transpile_expr(expr.right, env)
        return _eval_operand(expr.op, (left, right), env)
    if isinstance(expr, ast.UnaryOp):
        value = _transpile_expr(expr.operand, env)
        return _eval_operand(expr.op, (value,), env)
    if isinstance(expr, ast.Lambda):
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, ast.Compare):
        values = [expr.left] + expr.comparators
        if len(values) != 2:
            raise NotImplementedError('Comparison of 3 or more values is not implemented.')
        values = [_transpile_expr(e, env) for e in values]
        return _eval_operand(expr.ops[0], values, env)
    if isinstance(expr, ast.IfExp):
        cond = _transpile_expr(expr.test, env)
        x = _transpile_expr(expr.body, env)
        y = _transpile_expr(expr.orelse, env)
        if isinstance(expr, Constant):
            return x if expr.obj else y
        if cond.ctype.dtype.kind == 'c':
            raise TypeError('Complex type value cannot be boolean condition.')
        x, y = (_infer_type(x, y, env), _infer_type(y, x, env))
        if x.ctype.dtype != y.ctype.dtype:
            raise TypeError(f'Type mismatch in conditional expression.: {x.ctype.dtype} != {y.ctype.dtype}')
        cond = _astype_scalar(cond, _cuda_types.bool_, 'unsafe', env)
        return Data(f'({cond.code} ? {x.code} : {y.code})', x.ctype)
    if isinstance(expr, ast.Call):
        func = _transpile_expr(expr.func, env)
        args = [_transpile_expr(x, env) for x in expr.args]
        kwargs: Dict[str, Union[Constant, Data]] = {}
        for kw in expr.keywords:
            assert kw.arg is not None
            kwargs[kw.arg] = _transpile_expr(kw.value, env)
        builtin_funcs = _builtin_funcs.builtin_functions_dict
        if is_constants(func) and func.obj in builtin_funcs:
            func = builtin_funcs[func.obj]
        if isinstance(func, _internal_types.BuiltinFunc):
            return func.call(env, *args, **kwargs)
        if not isinstance(func, Constant):
            raise TypeError(f"'{func}' is not callable.")
        func = func.obj
        if isinstance(func, _interface._cuda_types.TypeBase):
            return func._instantiate(env, *args, **kwargs)
        if isinstance(func, _interface._JitRawKernel):
            if not func._device:
                raise TypeError(f'Calling __global__ function {func._func.__name__} from __global__ funcion is not allowed.')
            args = [Data.init(x, env) for x in args]
            in_types = tuple([x.ctype for x in args])
            fname, return_type = _transpile_func_obj(func._func, ['__device__'], env.mode, in_types, None, env.generated)
            in_params = ', '.join([x.code for x in args])
            return Data(f'{fname}({in_params})', return_type)
        if isinstance(func, _kernel.ufunc):
            dtype = kwargs.pop('dtype', Constant(None)).obj
            if len(kwargs) > 0:
                name = next(iter(kwargs))
                raise TypeError(f"'{name}' is an invalid keyword to ufunc {func.name}")
            return _call_ufunc(func, args, dtype, env)
        if is_constants(*args, *kwargs.values()):
            args = [x.obj for x in args]
            kwargs = dict([(k, v.obj) for k, v in kwargs.items()])
            return Constant(func(*args, **kwargs))
        if inspect.isclass(func) and issubclass(func, _typeclasses):
            if len(args) != 1:
                raise TypeError(f'function takes {func} invalid number of argument')
            ctype = _cuda_types.Scalar(func)
            return _astype_scalar(args[0], ctype, 'unsafe', env)
        raise TypeError(f"Invalid function call '{func.__name__}'.")
    if isinstance(expr, ast.Constant):
        return Constant(expr.value)
    if isinstance(expr, ast.Num):
        return Constant(expr.n)
    if isinstance(expr, ast.Str):
        return Constant(expr.s)
    if isinstance(expr, ast.NameConstant):
        return Constant(expr.value)
    if isinstance(expr, ast.Subscript):
        array = _transpile_expr(expr.value, env)
        index = _transpile_expr(expr.slice, env)
        return _indexing(array, index, env)
    if isinstance(expr, ast.Name):
        value = env[expr.id]
        if value is None:
            raise NameError(f'Unbound name: {expr.id}')
        return value
    if isinstance(expr, ast.Attribute):
        value = _transpile_expr(expr.value, env)
        if isinstance(value, Constant):
            return Constant(getattr(value.obj, expr.attr))
        if isinstance(value, Data) and hasattr(value.ctype, expr.attr):
            attr = getattr(value.ctype, expr.attr, None)
            if isinstance(attr, types.MethodType):
                return attr(value)
            if isinstance(attr, Data):
                return attr
        raise AttributeError(f'Unknown attribute: {expr.attr}')
    if isinstance(expr, ast.Tuple):
        elts = [_transpile_expr(x, env) for x in expr.elts]
        if all([isinstance(x, Constant) for x in elts]):
            return Constant(tuple([x.obj for x in elts]))
        elts = [Data.init(x, env) for x in elts]
        elts_code = ', '.join([x.code for x in elts])
        if len(elts) == 2:
            return Data(f'thrust::make_pair({elts_code})', _cuda_types.Tuple([x.ctype for x in elts]))
        else:
            return Data(f'thrust::make_tuple({elts_code})', _cuda_types.Tuple([x.ctype for x in elts]))
    if isinstance(expr, ast.Index):
        return _transpile_expr(expr.value, env)
    raise ValueError('Not supported: type {}'.format(type(expr)))