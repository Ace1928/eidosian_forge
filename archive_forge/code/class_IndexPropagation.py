import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
class IndexPropagation:
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(self, inner: Any):
        self._inner = inner

    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
        if isinstance(expr, sympy.Integer):
            return self._inner.constant(int(expr), dtype)
        elif expr.is_number:
            return self._inner.constant(float(expr), dtype)
        return self._inner.index_expr(expr, dtype)

    def unwrap(self, a: Union[Any, IndexPropVar]) -> Any:
        if isinstance(a, (list, tuple)):
            return tuple((self.unwrap(v) for v in a))
        if not isinstance(a, IndexPropVar):
            return a
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)
        return a.value

    def wrap(self, a) -> IndexPropResult:
        if isinstance(a, (list, tuple)):
            return tuple((self.wrap(v) for v in a))
        return IndexPropVar(a)

    @overload
    def fallback(self, name: Literal['indirect_indexing'], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> IndexPropVar:
        ...

    @overload
    def fallback(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> IndexPropResult:
        ...

    def fallback(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> IndexPropResult:
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return self.wrap(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> IndexPropResult:

        def unwrap(a: Union[Any, IndexPropVar]) -> Any:
            if not isinstance(a, IndexPropVar):
                return a
            return a.value
        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        is_valid_expr = new_expr is not NotImplemented and (isinstance(new_expr.expr, sympy.Number) or new_expr.expr.is_integer)
        if not is_valid_expr:
            return self.fallback(name, args, kwargs)
        return IndexPropVar.new_symbolic(new_expr)

    def __getattr__(self, name: str) -> Callable[..., IndexPropResult]:

        def inner(*args: Any, **kwargs: Any) -> IndexPropResult:
            if not hasattr(SymPyOps, name):
                return self.fallback(name, args, kwargs)
            var_arguments = [a for a in itertools.chain(args, kwargs.values()) if isinstance(a, IndexPropVar)]
            if not all((v.is_symbolic for v in var_arguments)):
                return self.fallback(name, args, kwargs)
            return self.propagate_sympy(name, args, kwargs)
        return inner

    def indirect_indexing(self, index: Union[Any, IndexPropVar], size: Any, check: bool=True) -> Any:
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            index = index.value.expr
            return index + Where(index >= 0, 0, size)
        return self.fallback('indirect_indexing', (index, size, check), {}).value