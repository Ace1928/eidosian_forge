import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
class SimulatesIntermediateState(Generic[TStepResult, TSimulationTrialResult, TSimulatorState], SimulatesFinalState[TSimulationTrialResult], metaclass=abc.ABCMeta):
    """A SimulatesFinalState that simulates a circuit by moments.

    Whereas a general SimulatesFinalState may return the entire simulator
    state at the end of a circuit, a SimulatesIntermediateState can
    simulate stepping through the moments of a circuit.

    Implementors of this interface should implement the _core_iterator
    method.

    Note that state here refers to simulator state, which is not necessarily
    a state vector.
    """

    def simulate_sweep_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TSimulationTrialResult]:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        state vector. In contrast to simulate, this allows for sweeping
        over different parameter values.

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
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        resolvers = list(study.to_resolvers(params))
        for i, param_resolver in enumerate(resolvers):
            state = initial_state.copy() if isinstance(initial_state, SimulationStateBase) and i < len(resolvers) - 1 else initial_state
            all_step_results = self.simulate_moment_steps(program, param_resolver, qubit_order, state)
            measurements: Dict[str, np.ndarray] = {}
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=np.uint8)
            yield self._create_simulator_trial_result(params=param_resolver, measurements=measurements, final_simulator_state=step_result._simulator_state())

    def simulate_moment_steps(self, circuit: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TStepResult]:
        """Returns an iterator of StepResults for each moment simulated.

        If the circuit being simulated is empty, a single step result should
        be returned with the state being set to the initial state.

        Args:
            circuit: The Circuit to simulate.
            param_resolver: A ParamResolver for determining values of Symbols.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. This can be
                either a raw state or a `TSimulationState`. The form of the
                raw state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            Iterator that steps through the simulation, simulating each
            moment and returning a StepResult for each moment.
        """
        param_resolver = study.ParamResolver(param_resolver)
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        actual_initial_state = 0 if initial_state is None else initial_state
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        return self._base_iterator(resolved_circuit, qubits, actual_initial_state)

    @abc.abstractmethod
    def _base_iterator(self, circuit: 'cirq.AbstractCircuit', qubits: Tuple['cirq.Qid', ...], initial_state: Any) -> Iterator[TStepResult]:
        """Iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            qubits: Specifies the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """

    @abc.abstractmethod
    def _create_simulator_trial_result(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: TSimulatorState) -> TSimulationTrialResult:
        """This method can be implemented to create a trial result.

        Args:
            params: The ParamResolver for this trial.
            measurements: The measurement results for this trial.
            final_simulator_state: The final state of the simulation.

        Returns:
            The SimulationTrialResult.
        """
        raise NotImplementedError()