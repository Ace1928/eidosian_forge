from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
class ExpressionMap(dict):
    """A dictionary with sympy expressions and symbols for keys and sympy
    symbols for values.

    This is returned by `cirq.flatten`.  See `ExpressionMap.transform_sweep` and
    `ExpressionMap.transform_params`.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the `ExpressionMap`.

        Takes the same arguments as the builtin `dict`.  Keys must be sympy
        expressions or symbols (instances of `sympy.Expr`).
        """
        super().__init__(*args, **kwargs)

    def transform_sweep(self, sweep: Union[sweeps.Sweep, List[resolver.ParamResolver]]) -> sweeps.Sweep:
        """Returns a sweep to use with a circuit flattened earlier with
        `cirq.flatten`.

        If `sweep` sweeps symbol `a` over (1.0, 2.0, 3.0) and this
        `ExpressionMap` maps `a/2+1` to the symbol `'<a/2 + 1>'` then this
        method returns a sweep that sweeps symbol `'<a/2 + 1>'` over
        (1.5, 2, 2.5).

        See `cirq.flatten` for an example.

        Args:
            sweep: The sweep to transform.
        """
        sweep = sweepable.to_sweep(sweep)
        param_list: List[resolver.ParamDictType] = []
        for r in sweep:
            param_dict: resolver.ParamDictType = {}
            for formula, sym in self.items():
                if isinstance(sym, (sympy.Symbol, str)):
                    param_dict[str(sym)] = protocols.resolve_parameters(formula, r)
            param_list.append(param_dict)
        return sweeps.ListSweep(param_list)

    def transform_params(self, params: resolver.ParamResolverOrSimilarType) -> resolver.ParamDictType:
        """Returns a `ParamResolver` to use with a circuit flattened earlier
        with `cirq.flatten`.

        If `params` maps symbol `a` to 3.0 and this `ExpressionMap` maps
        `a/2+1` to `'<a/2 + 1>'` then this method returns a resolver that maps
        symbol `'<a/2 + 1>'` to 2.5.

        See `cirq.flatten` for an example.

        Args:
            params: The params to transform.
        """
        param_dict: resolver.ParamDictType = {sym: protocols.resolve_parameters(formula, params) for formula, sym in self.items() if isinstance(sym, sympy.Expr)}
        return param_dict

    def __repr__(self) -> str:
        super_repr = super().__repr__()
        return f'cirq.ExpressionMap({super_repr})'