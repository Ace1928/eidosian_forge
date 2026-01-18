import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
class SymPyOps:
    """An ops handler where all IR values are SymPy expressions

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented

    """

    @staticmethod
    def identity(value: Any) -> Any:
        return value

    @staticmethod
    def constant(value: Union[int, float, bool], dtype: torch.dtype) -> TypedExpr:
        if is_boolean_dtype(dtype):
            expr = sympy.Integer(bool(value))
        elif is_integer_dtype(dtype):
            expr = sympy.Integer(int(value))
        else:
            expr = sympy.Float(float(value))
        return TypedExpr(expr, dtype)

    @staticmethod
    def index_expr(value: sympy.Expr, dtype: torch.dtype) -> Union[int, TypedExpr]:
        if isinstance(value, int):
            value = sympy.Integer(value)
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(value: Any, dtype: torch.dtype, src_dtype: Optional[torch.dtype]=None) -> Union[int, TypedExpr]:
        if isinstance(value.expr, (sympy.Integer, sympy.Float)):
            return SymPyOps.constant(value.expr, dtype)
        elif is_integer_dtype(dtype) and is_integer_dtype(value.dtype):
            return SymPyOps.index_expr(value.expr, dtype)
        else:
            return NotImplemented

    @staticmethod
    def square(x: TypedExpr) -> TypedExpr:
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr + y.expr, result_type)

    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr - y.expr, result_type)

    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr:
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented
        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> Optional[TypedExpr]:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented
        result_expr = ModularIndexing(x.expr, sympy.Integer(1), y.expr)
        return TypedExpr(result_expr, result_type)

    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(sympy.Min(x.expr, y.expr), result_type)

    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(sympy.Max(x.expr, y.expr), result_type)