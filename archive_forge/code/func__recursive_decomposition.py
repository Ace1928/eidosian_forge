from typing import List
from warnings import warn
import numpy as np
from scipy.sparse.linalg import expm as sparse_expm
import pennylane as qml
from pennylane import math
from pennylane.math import expand_matrix
from pennylane.operation import (
from pennylane.ops.qubit import Hamiltonian
from pennylane.wires import Wires
from .sprod import SProd
from .sum import Sum
from .symbolicop import ScalarSymbolicOp
def _recursive_decomposition(self, base: Operator, coeff: complex):
    """Decompose the exponential of ``base`` multiplied by ``coeff``.

        Args:
            base (Operator): exponentiated operator
            coeff (complex): coefficient multiplying the exponentiated operator

        Returns:
            List[Operator]: decomposition
        """
    if isinstance(base, Tensor) and len(base.wires) != len(base.obs):
        raise DecompositionUndefinedError(f'Unable to determine if the exponential has a decomposition when the base operator is a Tensor object with overlapping wires. Received base {base}.')
    if isinstance(base, Hamiltonian):
        base = qml.dot(base.coeffs, base.ops)
    elif isinstance(base, Tensor):
        base = qml.prod(*base.obs)
    if isinstance(base, SProd):
        return self._recursive_decomposition(base.base, base.scalar * coeff)
    if self.num_steps is not None and isinstance(base, (Hamiltonian, Sum)):
        coeffs = base.coeffs if isinstance(base, Hamiltonian) else [1] * len(base)
        coeffs = [c * coeff for c in coeffs]
        ops = base.ops if isinstance(base, Hamiltonian) else base.operands
        return self._trotter_decomposition(ops, coeffs)
    has_generator_types = []
    has_generator_types_anywires = []
    for op_name in qml.ops.qubit.__all__:
        op_class = getattr(qml.ops.qubit, op_name)
        if op_class.has_generator:
            if op_class.num_wires == AnyWires:
                has_generator_types_anywires.append(op_class)
            elif op_class.num_wires == len(base.wires):
                has_generator_types.append(op_class)
    has_generator_types.extend(has_generator_types_anywires)
    for op_class in has_generator_types:
        if op_class not in {qml.PauliRot, qml.PCPhase}:
            g, c = qml.generator(op_class)(coeff, base.wires)
            mapped_wires_g = qml.map_wires(g, dict(zip(g.wires, base.wires)))
            if qml.equal(base, mapped_wires_g) and math.real(coeff) == 0:
                coeff = math.real(-1j / c * coeff)
                return [op_class(coeff, g.wires)]
            simplified_g = qml.simplify(qml.s_prod(c, mapped_wires_g))
            if qml.equal(base, simplified_g) and math.real(coeff) == 0:
                coeff = math.real(-1j * coeff)
                return [op_class(coeff, g.wires)]
    if qml.pauli.is_pauli_word(base) and math.real(coeff) == 0:
        return self._pauli_rot_decomposition(base, coeff)
    error_msg = f'The decomposition of the {self} operator is not defined. '
    if not self.num_steps:
        error_msg += 'Please set a value to ``num_steps`` when instantiating the ``Exp`` operator if a Suzuki-Trotter decomposition is required. '
    if math.real(self.coeff) != 0 and self.base.is_hermitian:
        error_msg += 'Decomposition is not defined for real coefficients of hermitian operators.'
    raise DecompositionUndefinedError(error_msg)