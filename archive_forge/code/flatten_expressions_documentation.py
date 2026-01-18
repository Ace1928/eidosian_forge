from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
Returns a `ParamResolver` to use with a circuit flattened earlier
        with `cirq.flatten`.

        If `params` maps symbol `a` to 3.0 and this `ExpressionMap` maps
        `a/2+1` to `'<a/2 + 1>'` then this method returns a resolver that maps
        symbol `'<a/2 + 1>'` to 2.5.

        See `cirq.flatten` for an example.

        Args:
            params: The params to transform.
        