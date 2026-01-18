import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
@value.value_equality
class _MPSHandler(qis.QuantumStateRepresentation):
    """Quantum state of the MPS simulation."""

    def __init__(self, qid_shape: Tuple[int, ...], grouping: Dict[int, int], M: List[qtn.Tensor], format_i: str, estimated_gate_error_list: List[float], simulation_options: MPSOptions=MPSOptions()):
        """Creates an MPSQuantumState

        Args:
            qid_shape: Dimensions of the qubits represented.
            grouping: How to group qubits together, if None all are individual.
            M: The tensor list for maintaining the MPS state.
            format_i: A string for formatting the group labels.
            estimated_gate_error_list: The error estimations.
            simulation_options: Numerical options for the simulation.
        """
        self._qid_shape = qid_shape
        self._grouping = grouping
        self._M = M
        self._format_i = format_i
        self._format_mu = 'mu_{}_{}'
        self._simulation_options = simulation_options
        self._estimated_gate_error_list = estimated_gate_error_list

    @classmethod
    def create(cls, *, qid_shape: Tuple[int, ...], grouping: Dict[int, int], initial_state: int=0, simulation_options: MPSOptions=MPSOptions()):
        """Creates an MPSQuantumState

        Args:
            qid_shape: Dimensions of the qubits represented.
            grouping: How to group qubits together, if None all are individual.
            initial_state: The initial computational basis state.
            simulation_options: Numerical options for the simulation.

        Raises:
            ValueError: If the grouping does not cover the qubits.
        """
        M = []
        for _ in range(max(grouping.values()) + 1):
            M.append(qtn.Tensor())
        max_num_digits = len(f'{max(grouping.values())}')
        format_i = f'i_{{:0{max_num_digits}}}'
        for axis in reversed(range(len(qid_shape))):
            d = qid_shape[axis]
            x = np.zeros(d)
            x[initial_state % d] = 1.0
            n = grouping[axis]
            M[n] @= qtn.Tensor(x, inds=(format_i.format(axis),))
            initial_state = initial_state // d
        return _MPSHandler(qid_shape=qid_shape, grouping=grouping, M=M, format_i=format_i, estimated_gate_error_list=[], simulation_options=simulation_options)

    def i_str(self, i: int) -> str:
        return self._format_i.format(i)

    def mu_str(self, i: int, j: int) -> str:
        smallest = min(i, j)
        largest = max(i, j)
        return self._format_mu.format(smallest, largest)

    def __str__(self) -> str:
        return str(qtn.TensorNetwork(self._M))

    def _value_equality_values_(self) -> Any:
        return (self._qid_shape, self._M, self._simulation_options, self._grouping)

    def copy(self, deep_copy_buffers: bool=True) -> '_MPSHandler':
        """Copies the object.

        Args:
            deep_copy_buffers: True by default, False to reuse the existing buffers.
        Returns:
            A copy of the object.
        """
        return _MPSHandler(simulation_options=self._simulation_options, grouping=self._grouping, qid_shape=self._qid_shape, M=[x.copy() for x in self._M], estimated_gate_error_list=self._estimated_gate_error_list.copy(), format_i=self._format_i)

    def state_vector(self) -> np.ndarray:
        """Returns the full state vector.

        Returns:
            A vector that contains the full state.
        """
        tensor_network = qtn.TensorNetwork(self._M)
        state_vector = tensor_network.contract(inplace=False)
        sorted_ind = tuple(sorted(state_vector.inds))
        return state_vector.fuse({'i': sorted_ind}).data

    def partial_trace(self, keep_axes: Set[int]) -> np.ndarray:
        """Traces out all qubits except keep_axes.

        Args:
            keep_axes: The set of axes that are left after computing the
                partial trace. For example, if we have a circuit for 3 qubits
                and this parameter only has one qubit, the entire density matrix
                would be 8x8, but this function returns a 2x2 matrix.

        Returns:
            An array that contains the partial trace.
        """
        contracted_inds = set(map(self.i_str, set(range(len(self._qid_shape))) - keep_axes))
        conj_pfx = 'conj_'
        tensor_network = qtn.TensorNetwork(self._M)
        conj_tensor_network = tensor_network.conj()
        reindex_mapping = {}
        for M in conj_tensor_network.tensors:
            for ind in M.inds:
                if ind not in contracted_inds:
                    reindex_mapping[ind] = conj_pfx + ind
        conj_tensor_network.reindex(reindex_mapping, inplace=True)
        partial_trace = conj_tensor_network @ tensor_network
        forward_inds = list(map(self.i_str, keep_axes))
        backward_inds = [conj_pfx + forward_ind for forward_ind in forward_inds]
        return partial_trace.to_dense(forward_inds, backward_inds)

    def to_numpy(self) -> np.ndarray:
        """An alias for the state vector."""
        return self.state_vector()

    def apply_op(self, op: Any, axes: Sequence[int], prng: np.random.RandomState):
        """Applies a unitary operation, mutating the object to represent the new state.

        op:
            The operation that mutates the object. Note that currently, only 1-
            and 2- qubit operations are currently supported.
        """
        old_inds = tuple(map(self.i_str, axes))
        new_inds = tuple(['new_' + old_ind for old_ind in old_inds])
        if protocols.has_unitary(op):
            U = protocols.unitary(op)
        else:
            mixtures = protocols.mixture(op)
            mixture_idx = int(prng.choice(len(mixtures), p=[mixture[0] for mixture in mixtures]))
            U = mixtures[mixture_idx][1]
        U = qtn.Tensor(U.reshape([self._qid_shape[axis] for axis in axes] * 2), inds=new_inds + old_inds)
        if len(axes) == 1:
            n = self._grouping[axes[0]]
            self._M[n] = (U @ self._M[n]).reindex({new_inds[0]: old_inds[0]})
        elif len(axes) == 2:
            n, p = [self._grouping[axis] for axis in axes]
            if n == p:
                self._M[n] = (U @ self._M[n]).reindex({new_inds[0]: old_inds[0], new_inds[1]: old_inds[1]})
            else:
                mu_ind = self.mu_str(n, p)
                if mu_ind not in self._M[n].inds:
                    self._M[n].new_ind(mu_ind)
                if mu_ind not in self._M[p].inds:
                    self._M[p].new_ind(mu_ind)
                T = U @ self._M[n] @ self._M[p]
                left_inds = tuple(set(T.inds) & set(self._M[n].inds)) + (new_inds[0],)
                X, Y = T.split(left_inds, method=self._simulation_options.method, max_bond=self._simulation_options.max_bond, cutoff=self._simulation_options.cutoff, cutoff_mode=self._simulation_options.cutoff_mode, get='tensors', absorb='both', bond_ind=mu_ind)
                e_n = self._simulation_options.cutoff
                self._estimated_gate_error_list.append(e_n)
                self._M[n] = X.reindex({new_inds[0]: old_inds[0]})
                self._M[p] = Y.reindex({new_inds[1]: old_inds[1]})
        else:
            raise ValueError('Can only handle 1 and 2 qubit operations')
        return True

    def estimation_stats(self):
        """Returns some statistics about the memory usage and quality of the approximation."""
        num_coefs_used = sum([Mi.data.size for Mi in self._M])
        memory_bytes = sum([Mi.data.nbytes for Mi in self._M])
        estimated_fidelity = 1.0 + np.expm1(sum((np.log1p(-x) for x in self._estimated_gate_error_list)))
        estimated_fidelity = round(estimated_fidelity, ndigits=3)
        return {'num_coefs_used': num_coefs_used, 'memory_bytes': memory_bytes, 'estimated_fidelity': estimated_fidelity}

    def _measure(self, axes: Sequence[int], prng: np.random.RandomState, collapse_state_vector=True) -> List[int]:
        results: List[int] = []
        if collapse_state_vector:
            state = self
        else:
            state = self.copy()
        for axis in axes:
            M = state.partial_trace(keep_axes={axis})
            probs = np.diag(M).real
            sum_probs = sum(probs)
            if abs(sum_probs - 1.0) > self._simulation_options.sum_prob_atol:
                raise ValueError(f'Sum of probabilities exceeds tolerance: {sum_probs}')
            norm_probs = [x / sum_probs for x in probs]
            d = self._qid_shape[axis]
            result: int = int(prng.choice(d, p=norm_probs))
            collapser = np.zeros((d, d))
            collapser[result][result] = 1.0 / math.sqrt(probs[result])
            old_n = state.i_str(axis)
            new_n = 'new_' + old_n
            collapser = qtn.Tensor(collapser, inds=(new_n, old_n))
            state._M[axis] = (collapser @ state._M[axis]).reindex({new_n: old_n})
            results.append(result)
        return results

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[int]:
        """Measures the MPS.

        Args:
            axes: The axes to measure.
            seed: The random number seed to use.
        Returns:
            The measurements in axis order.
        """
        return self._measure(axes, value.parse_random_state(seed))

    def sample(self, axes: Sequence[int], repetitions: int=1, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
        """Samples the MPS.

        Args:
            axes: The axes to sample.
            repetitions: The number of samples to make.
            seed: The random number seed to use.
        Returns:
            The samples in order.
        """
        measurements: List[List[int]] = []
        prng = value.parse_random_state(seed)
        for _ in range(repetitions):
            measurements.append(self._measure(axes, prng, collapse_state_vector=False))
        return np.array(measurements, dtype=int)