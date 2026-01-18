import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
class DensePauliString(BaseDensePauliString):
    """An immutable string of Paulis, like `XIXY`, with a coefficient.

    A `DensePauliString` represents a multi-qubit pauli operator, i.e. a tensor product of single
    qubits Pauli gates (including the `cirq.IdentityGate`), each of which would act on a
    different qubit. When applied on qubits, a `DensePauliString` results in `cirq.PauliString`
    as an operation.

    Note that `cirq.PauliString` only stores a tensor product of non-identity `cirq.Pauli`
    operations whereas `cirq.DensePauliString` also supports storing the `cirq.IdentityGate`.

    For example,

    >>> dps = cirq.DensePauliString('XXIY')
    >>> print(dps) # 4 qubit pauli operator with 'X' on first 2 qubits, 'I' on 3rd and 'Y' on 4th.
    +XXIY
    >>> ps = dps.on(*cirq.LineQubit.range(4)) # When applied on qubits, we get a `cirq.PauliString`.
    >>> print(ps) # Note that `cirq.PauliString` only preserves non-identity operations.
    X(q(0))*X(q(1))*Y(q(3))

    This can optionally take a coefficient, for example:

    >>> dps = cirq.DensePauliString("XX", coefficient=3)
    >>> print(dps) # Represents 3 times the operator XX acting on two qubits.
    (3+0j)*XX
    >>> print(dps.on(*cirq.LineQubit.range(2))) # Coefficient is propagated to `cirq.PauliString`.
    (3+0j)*X(q(0))*X(q(1))

    If the coefficient has magnitude of 1, the resulting operator is a unitary and thus is
    also a `cirq.Gate`.

    Note that `DensePauliString` is an immutable object. If you need a mutable version of
    dense pauli strings, see `cirq.MutableDensePauliString`.
    """

    def frozen(self) -> 'DensePauliString':
        return self

    def copy(self, coefficient: Optional[Union[sympy.Expr, int, float, complex]]=None, pauli_mask: Union[None, str, Iterable[int], np.ndarray]=None) -> 'DensePauliString':
        if pauli_mask is None and (coefficient is None or coefficient == self.coefficient):
            return self
        return DensePauliString(coefficient=self.coefficient if coefficient is None else coefficient, pauli_mask=self.pauli_mask if pauli_mask is None else pauli_mask)