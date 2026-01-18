from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor, _to_superop
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
def _evolve(self, state, qargs=None):
    """Evolve a quantum state by the quantum channel.

        Args:
            state (DensityMatrix or Statevector): The input state.
            qargs (list): a list of quantum state subsystem positions to apply
                           the quantum channel on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.

        Raises:
            QiskitError: if the quantum channel dimension does not match the
                         specified quantum state subsystem dimensions.
        """
    from qiskit.quantum_info.states.densitymatrix import DensityMatrix
    if not isinstance(state, DensityMatrix):
        state = DensityMatrix(state)
    if qargs is None:
        if state._op_shape.shape[0] != self._op_shape.shape[1]:
            raise QiskitError('Operator input dimension is not equal to density matrix dimension.')
        vec = np.ravel(state.data, order='F')
        mat = np.reshape(np.dot(self.data, vec), (self._output_dim, self._output_dim), order='F')
        return DensityMatrix(mat, dims=self.output_dims())
    if state.dims(qargs) != self.input_dims():
        raise QiskitError('Operator input dimensions are not equal to statevector subsystem dimensions.')
    tensor = np.reshape(state.data, state._op_shape.tensor_shape)
    mat = np.reshape(self.data, self._tensor_shape)
    num_indices = len(state.dims())
    indices = [num_indices - 1 - qubit for qubit in qargs] + [2 * num_indices - 1 - qubit for qubit in qargs]
    tensor = Operator._einsum_matmul(tensor, mat, indices)
    new_dims = list(state.dims())
    output_dims = self.output_dims()
    for i, qubit in enumerate(qargs):
        new_dims[qubit] = output_dims[i]
    new_dim = np.prod(new_dims)
    tensor = np.reshape(tensor, (new_dim, new_dim))
    return DensityMatrix(tensor, dims=new_dims)