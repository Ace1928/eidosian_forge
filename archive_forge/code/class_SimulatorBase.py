import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
class SimulatorBase(Generic[TStepResultBase, TSimulationTrialResult, TSimulationState], SimulatesIntermediateState[TStepResultBase, TSimulationTrialResult, SimulationStateBase[TSimulationState]], SimulatesSamples, metaclass=abc.ABCMeta):
    """A base class for the built-in simulators.

    Most implementors of this interface should implement the
    `_create_partial_simulation_state` and `_create_step_result` methods. The
    first one creates the simulator's quantum state representation at the
    beginning of the simulation. The second creates the step result emitted
    after each `Moment` in the simulation.

    Iteration in the subclass is handled by the `_core_iterator` implementation
    here, which handles moment stepping, application of operations, measurement
    collection, and creation of noise. Simulators with more advanced needs can
    override the implementation if necessary.

    Sampling is handled by the implementation of `_run`. This implementation
    iterates the circuit to create a final step result, and samples that
    result when possible. If not possible, due to noise or classical
    probabilities on a state vector, the implementation attempts to fully
    iterate the unitary prefix once, then only repeat the non-unitary
    suffix from copies of the state obtained by the prefix. If more advanced
    functionality is required, then the `_run` method can be overridden.

    Note that state here refers to simulator state, which is not necessarily
    a state vector. The included simulators and corresponding states are state
    vector, density matrix, Clifford, and MPS. Each of these use the default
    `_core_iterator` and `_run` methods.
    """

    def __init__(self, *, dtype: Type[np.complexfloating]=np.complex64, noise: 'cirq.NOISE_MODEL_LIKE'=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, split_untangled_states: bool=False):
        """Initializes the simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation.
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            split_untangled_states: If True, optimizes simulation by running
                unentangled qubit sets independently and merging those states
                at the end.
        """
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)
        self._noise = devices.NoiseModel.from_noise_model_like(noise)
        self._split_untangled_states = split_untangled_states

    @property
    def noise(self) -> 'cirq.NoiseModel':
        return self._noise

    @abc.abstractmethod
    def _create_partial_simulation_state(self, initial_state: Any, qubits: Sequence['cirq.Qid'], classical_data: 'cirq.ClassicalDataStore') -> TSimulationState:
        """Creates an instance of the TSimulationState class for the simulator.

        It represents the supplied qubits initialized to the provided state.

        Args:
            initial_state: The initial state to represent. An integer state is
                understood to be a pure state. Other state representations are
                simulator-dependent.
            qubits: The sequence of qubits to represent.
            classical_data: The shared classical data container for this
                simulation.
        """

    @abc.abstractmethod
    def _create_step_result(self, sim_state: SimulationStateBase[TSimulationState]) -> TStepResultBase:
        """This method should be implemented to create a step result.

        Args:
            sim_state: The SimulationStateBase for this trial.

        Returns:
            The StepResult.
        """

    def _can_be_in_run_prefix(self, val: Any):
        """Determines what should be put in the prefix in `_run`

        The `_run` method has an optimization that reduces repetition by
        splitting the circuit into a prefix that is pure with respect to the
        state representation, and only executing that once per sample set. For
        state vectors, any unitary operation is pure, and we make this the
        default here. For density matrices, any non-measurement operation can
        be represented wholely in the matrix, and thus this method is
        overridden there to enable greater optimization there.

        Custom simulators can override this method appropriately.

        Args:
            val: An operation or noise model to test for purity within the
                state representation.

        Returns:
            A boolean representing whether the value can be added to the
            `_run` prefix."""
        return protocols.has_unitary(val)

    def _base_iterator(self, circuit: 'cirq.AbstractCircuit', qubits: Tuple['cirq.Qid', ...], initial_state: Any) -> Iterator[TStepResultBase]:
        sim_state = self._create_simulation_state(initial_state, qubits)
        return self._core_iterator(circuit, sim_state)

    def _core_iterator(self, circuit: 'cirq.AbstractCircuit', sim_state: SimulationStateBase[TSimulationState], all_measurements_are_terminal: bool=False) -> Iterator[TStepResultBase]:
        """Standard iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            sim_state: The initial args for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            all_measurements_are_terminal: Whether all measurements in the
                given circuit are terminal.

        Yields:
            StepResults from simulating a Moment of the Circuit.

        Raises:
            TypeError: The simulator encounters an op it does not support.
        """
        if len(circuit) == 0:
            yield self._create_step_result(sim_state)
            return
        noisy_moments = self.noise.noisy_moments(circuit, sorted(circuit.all_qubits()))
        measured: Dict[Tuple['cirq.Qid', ...], bool] = collections.defaultdict(bool)
        for moment in noisy_moments:
            for op in ops.flatten_to_ops(moment):
                try:
                    if all_measurements_are_terminal and measured[op.qubits]:
                        continue
                    if isinstance(op.gate, ops.MeasurementGate):
                        measured[op.qubits] = True
                        if all_measurements_are_terminal:
                            continue
                    protocols.act_on(op, sim_state)
                except TypeError:
                    raise TypeError(f"{self.__class__.__name__} doesn't support {op!r}")
            yield self._create_step_result(sim_state)

    def _run(self, circuit: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolver', repetitions: int) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        qubits = tuple(sorted(resolved_circuit.all_qubits()))
        sim_state = self._create_simulation_state(0, qubits)
        prefix, general_suffix = split_into_matching_protocol_then_general(resolved_circuit, self._can_be_in_run_prefix) if self._can_be_in_run_prefix(self.noise) else (resolved_circuit[0:0], resolved_circuit)
        step_result = None
        for step_result in self._core_iterator(circuit=prefix, sim_state=sim_state):
            pass
        general_ops = list(general_suffix.all_operations())
        if all((isinstance(op.gate, ops.MeasurementGate) for op in general_ops)):
            for step_result in self._core_iterator(circuit=general_suffix, sim_state=sim_state, all_measurements_are_terminal=True):
                pass
            assert step_result is not None
            measurement_ops = [cast(ops.GateOperation, op) for op in general_ops]
            return step_result.sample_measurement_ops(measurement_ops, repetitions, seed=self._prng, _allow_repeated=True)
        records: Dict['cirq.MeasurementKey', List[Sequence[Sequence[int]]]] = {}
        for i in range(repetitions):
            for step_result in self._core_iterator(general_suffix, sim_state=sim_state.copy(deep_copy_buffers=False) if i < repetitions - 1 else sim_state):
                pass
            for k, r in step_result._classical_data.records.items():
                if k not in records:
                    records[k] = []
                records[k].append(r)
            for k, cr in step_result._classical_data.channel_records.items():
                if k not in records:
                    records[k] = []
                records[k].append([cr])

        def pad_evenly(results: Sequence[Sequence[Sequence[int]]]):
            largest = max((len(result) for result in results))
            xs = np.zeros((len(results), largest, len(results[0][0])), dtype=np.uint8)
            for i, result in enumerate(results):
                xs[i, 0:len(result), :] = result
            return xs
        return {str(k): pad_evenly(v) for k, v in records.items()}

    def simulate_sweep_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TSimulationTrialResult]:
        """Simulates the supplied Circuit.

        This particular implementation overrides the base implementation such
        that an unparameterized prefix circuit is simulated and fed into the
        parameterized suffix circuit.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. This can be
                either a raw state or an `SimulationStateBase`. The form of the
                raw state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """

        def sweep_prefixable(op: 'cirq.Operation'):
            return self._can_be_in_run_prefix(op) and (not protocols.is_parameterized(op))
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(program.all_qubits())
        initial_state = 0 if initial_state is None else initial_state
        sim_state = self._create_simulation_state(initial_state, qubits)
        prefix, suffix = split_into_matching_protocol_then_general(program, sweep_prefixable) if self._can_be_in_run_prefix(self.noise) else (program[0:0], program)
        step_result = None
        for step_result in self._core_iterator(circuit=prefix, sim_state=sim_state):
            pass
        sim_state = step_result._sim_state
        yield from super().simulate_sweep_iter(suffix, params, qubit_order, sim_state)

    def _create_simulation_state(self, initial_state: Any, qubits: Sequence['cirq.Qid']) -> SimulationStateBase[TSimulationState]:
        if isinstance(initial_state, SimulationStateBase):
            return initial_state
        classical_data = value.ClassicalDataDictionaryStore()
        if self._split_untangled_states:
            args_map: Dict[Optional['cirq.Qid'], TSimulationState] = {}
            if isinstance(initial_state, int):
                for q in reversed(qubits):
                    args_map[q] = self._create_partial_simulation_state(initial_state=initial_state % q.dimension, qubits=[q], classical_data=classical_data)
                    initial_state = int(initial_state / q.dimension)
            else:
                args = self._create_partial_simulation_state(initial_state=initial_state, qubits=qubits, classical_data=classical_data)
                for q in qubits:
                    args_map[q] = args
            args_map[None] = self._create_partial_simulation_state(0, (), classical_data)
            return SimulationProductState(args_map, qubits, self._split_untangled_states, classical_data=classical_data)
        else:
            return self._create_partial_simulation_state(initial_state=initial_state, qubits=qubits, classical_data=classical_data)