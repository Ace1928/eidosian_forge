from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
Converts, if possible, the given `gate` to an equivalent instance of `target_gate_type`.

        This method can be used for converting instances of `POSSIBLE_FSIM_GATES` to other
        equivalent types from the same group. For example, you can convert a sqrt iswap gate
        to an equivalent fsim gate by calling:

        >>> gf = cirq_google.FSimGateFamily()
        >>> assert gf.convert(cirq.SQRT_ISWAP, cirq.FSimGate) == cirq.FSimGate(-np.pi/4, 0)

        The method can also be used for converting parameterized gate instances, by setting
        `allow_symbols=True` in the gate family constructor. Note that, conversion of
        parameterized gate instances tries to be lenient and assumes that the correct
        parameters would eventually be filled during parameter resolution. This can also result
        in dropping extra parameters during type conversion, assuming the dropped parameters
        would be supplied the correct values. For example:

        >>> gf = cirq_google.FSimGateFamily(allow_symbols = True)
        >>> theta, phi = sympy.Symbol("theta"), sympy.Symbol("phi")
        >>> assert gf.convert(cirq.FSimGate(-np.pi/4, phi), cirq.ISwapPowGate) == cirq.SQRT_ISWAP
        >>> assert gf.convert(cirq.FSimGate(theta, np.pi/4), cirq.ISwapPowGate) is None

        Args:
            gate            : `cirq.Gate` instance to convert.
            target_gate_type: One of `POSSIBLE_FSIM_GATES` types to which the given gate should be
                              converted to.
        Returns:
            The converted gate instances if the conversion is possible, else None.
        Raises:
            ValueError: If `target_gate_type` is not one of `POSSIBLE_FSIM_GATES`.
        