import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
class SimulatesAmplitudes(metaclass=value.ABCMetaImplementAnyOneOf):
    """Simulator that computes final amplitudes of given bitstrings.

    Given a circuit and a list of bitstrings, computes the amplitudes
    of the given bitstrings in the state obtained by applying the circuit
    to the all zeros state. Implementors of this interface should implement
    the compute_amplitudes_sweep_iter method.
    """

    def compute_amplitudes(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Sequence[complex]:
        """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering. If inputting
                a binary literal add the prefix 0b or 0B.
                For example: 0010 can be input as 0b0010, 0B0010, 2, 0x2, etc.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            List of amplitudes.
        """
        return self.compute_amplitudes_sweep(program, bitstrings, study.ParamResolver(param_resolver), qubit_order)[0]

    def compute_amplitudes_sweep(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Sequence[Sequence[complex]]:
        """Wraps computed amplitudes in a list.

        Prefer overriding `compute_amplitudes_sweep_iter`.
        """
        return list(self.compute_amplitudes_sweep_iter(program, bitstrings, params, qubit_order))

    def _compute_amplitudes_sweep_to_iter(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Iterator[Sequence[complex]]:
        if type(self).compute_amplitudes_sweep == SimulatesAmplitudes.compute_amplitudes_sweep:
            raise RecursionError('Must define either compute_amplitudes_sweep or compute_amplitudes_sweep_iter.')
        yield from self.compute_amplitudes_sweep(program, bitstrings, params, qubit_order)

    @value.alternative(requires='compute_amplitudes_sweep', implementation=_compute_amplitudes_sweep_to_iter)
    def compute_amplitudes_sweep_iter(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Iterator[Sequence[complex]]:
        """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering. If inputting
                a binary literal add the prefix 0b or 0B.
                For example: 0010 can be input as 0b0010, 0B0010, 2, 0x2, etc.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            An Iterator over lists of amplitudes. The outer dimension indexes
            the circuit parameters and the inner dimension indexes bitstrings.
        """
        raise NotImplementedError()

    def sample_from_amplitudes(self, circuit: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolver', seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE', repetitions: int=1, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Dict[int, int]:
        """Uses amplitude simulation to sample from the given circuit.

        This implements the algorithm outlined by Bravyi, Gosset, and Liu in
        https://arxiv.org/abs/2112.08499 to more efficiently calculate samples
        given an amplitude-based simulator.

        Simulators which also implement SimulatesSamples or SimulatesFullState
        should prefer `run()` or `simulate()`, respectively, as this method
        only accelerates sampling for amplitude-based simulators.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            seed: Random state to use as a seed. This must be provided
                manually - if the simulator has its own seed, it will not be
                used unless it is passed as this argument.
            repetitions: The number of repetitions to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            A dict of bitstrings sampled from the final state of `circuit` to
            the number of occurrences of that bitstring.

        Raises:
            ValueError: if 'circuit' has non-unitary elements, as differences
                in behavior between sampling steps break this algorithm.
        """
        prng = value.parse_random_state(seed)
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        base_circuit = circuits.Circuit((ops.I(q) for q in qubits)) + circuit.unfreeze()
        qmap = {q: i for i, q in enumerate(qubits)}
        current_samples = {(0,) * len(qubits): repetitions}
        solved_circuit = protocols.resolve_parameters(base_circuit, param_resolver)
        if not protocols.has_unitary(solved_circuit):
            raise ValueError('sample_from_amplitudes does not support non-unitary behavior.')
        if protocols.is_measurement(solved_circuit):
            raise ValueError('sample_from_amplitudes does not support intermediate measurement.')
        for m_id, moment in enumerate(solved_circuit[1:]):
            circuit_prefix = solved_circuit[:m_id + 1]
            for t, op in enumerate(moment.operations):
                new_samples: Dict[Tuple[int, ...], int] = collections.defaultdict(int)
                qubit_indices = {qmap[q] for q in op.qubits}
                subcircuit = circuit_prefix + circuits.Moment(moment.operations[:t + 1])
                for current_sample, count in current_samples.items():
                    sample_set = [current_sample]
                    for idx in qubit_indices:
                        sample_set = [target[:idx] + (result,) + target[idx + 1:] for target in sample_set for result in [0, 1]]
                    bitstrings = [int(''.join(map(str, sample)), base=2) for sample in sample_set]
                    amps = self.compute_amplitudes(subcircuit, bitstrings, qubit_order=qubit_order)
                    weights = np.abs(np.square(np.array(amps))).astype(np.float64)
                    weights /= np.linalg.norm(weights, 1)
                    subsample = prng.choice(len(sample_set), p=weights, size=count)
                    for sample_index in subsample:
                        new_samples[sample_set[sample_index]] += 1
                current_samples = new_samples
        return {int(''.join(map(str, k)), base=2): v for k, v in current_samples.items()}