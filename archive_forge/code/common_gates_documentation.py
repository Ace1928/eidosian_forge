from typing import (
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import controlled_gate, eigen_gate, gate_features, raw_types, control_values as cv
from cirq.type_workarounds import NotImplementedType
from cirq.ops.swap_gates import ISWAP, SWAP, ISwapPowGate, SwapPowGate
from cirq.ops.measurement_gate import MeasurementGate
imports.
Returns a controlled `CXPowGate`, using a `CCXPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CCXPowGate` or a `ControlledGate` of a `CCXPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `CXPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CCXPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CCXPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CCXPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CCXPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CCXPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        