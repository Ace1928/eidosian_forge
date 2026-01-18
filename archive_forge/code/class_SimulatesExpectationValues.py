import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
class SimulatesExpectationValues(metaclass=value.ABCMetaImplementAnyOneOf):
    """Simulator that computes exact expectation values of observables.

    Given a circuit and an observable map, computes exact (to float precision)
    expectation values for each observable at the end of the circuit.

    Implementors of this interface should implement the
    simulate_expectation_values_sweep_iter method.
    """

    def simulate_expectation_values(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> List[float]:
        """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state.

        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.

        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            permit_terminal_measurements: If the provided circuit ends with
                measurement(s), this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.

        Returns:
            A list of expectation values, with the value at index `n`
            corresponding to `observables[n]` from the input.

        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False.
        """
        return self.simulate_expectation_values_sweep(program, observables, study.ParamResolver(param_resolver), qubit_order, initial_state, permit_terminal_measurements)[0]

    def simulate_expectation_values_sweep(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> List[List[float]]:
        """Wraps computed expectation values in a list.

        Prefer overriding `simulate_expectation_values_sweep_iter`.
        """
        return list(self.simulate_expectation_values_sweep_iter(program, observables, params, qubit_order, initial_state, permit_terminal_measurements))

    def _simulate_expectation_values_sweep_to_iter(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> Iterator[List[float]]:
        if type(self).simulate_expectation_values_sweep == SimulatesExpectationValues.simulate_expectation_values_sweep:
            raise RecursionError('Must define either simulate_expectation_values_sweep or simulate_expectation_values_sweep_iter.')
        yield from self.simulate_expectation_values_sweep(program, observables, params, qubit_order, initial_state, permit_terminal_measurements)

    @value.alternative(requires='simulate_expectation_values_sweep', implementation=_simulate_expectation_values_sweep_to_iter)
    def simulate_expectation_values_sweep_iter(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> Iterator[List[float]]:
        """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state, sweeping over the
        given params.

        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.

        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            permit_terminal_measurements: If the provided circuit ends in a
                measurement, this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.

        Returns:
            An Iterator over expectation-value lists. The outer index determines
            the sweep, and the inner index determines the observable. For
            instance, results[1][3] would select the fourth observable measured
            in the second sweep.

        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False.
        """
        raise NotImplementedError