from typing import Any, Dict, List, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import ops, protocols, study, value
from cirq._compat import proper_repr
from cirq.sim import simulator, density_matrix_simulation_state, simulator_base
class DensityMatrixSimulator(simulator_base.SimulatorBase['cirq.DensityMatrixStepResult', 'cirq.DensityMatrixTrialResult', 'cirq.DensityMatrixSimulationState'], simulator.SimulatesExpectationValues):
    """A simulator for density matrices and noisy quantum circuits.

    This simulator can be applied on circuits that are made up of operations
    that have:
        * a `_kraus_` method for a Kraus representation of a quantum channel.
        * a `_mixture_` method for a probabilistic combination of unitary gates.
        * a `_unitary_` method for a unitary gate.
        * a `_has_unitary_` and `_apply_unitary_` method.
        * measurements
        * a `_decompose_` that eventually yields one of the above
    That is, the circuit must have elements that follow on of the protocols:
        * `cirq.SupportsKraus`
        * `cirq.SupportsMixture`
        * `cirq.SupportsConsistentApplyUnitary`
        * `cirq.SupportsUnitary`
        * `cirq.SupportsDecompose`
    or is a measurement.

    This simulator supports three types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the density matrix (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    These methods return `Result`s which contain both the measurement
    results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the density
    matrix of the simulation at the end of the simulation of the circuit.
    Note that if the circuit contains measurements then the density matrix
    is that result for those particular measurement results. For example
    if there is one measurement, then the simulation may result in the
    measurement result for this measurement, and the density matrix will
    be that conditional on that result. It will not be the density matrix formed
    by summing over the different measurements and their probabilities.
    The simulate methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `DensityMatrixTrialResult` for details of this ordering). The initial
    state can be either the full density matrix, the full wave function (for
    pure states), or an integer which represents the initial state of being
    in a computational basis state for the binary representation of that
    integer. Similar to run methods, there are two simulate methods that run
    for single simulations or for sweeps across different parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a
    `DensityMatrixTrialResult` which contains, in addition to measurement
    results and information about the parameters that were used in the
    simulation, access to the density matrix via the `density_matrix` method.

    If one wishes to perform simulations that have access to the
    density matrix as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the density matrix.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments with the following
    (replace 'sim' with your `Simulator` object):

        for step_result in sim.simulate_moment_steps(circuit):
           # do something with the density matrix via
           # step_result.density_matrix()
    """

    def __init__(self, *, dtype: Type[np.complexfloating]=np.complex64, noise: 'cirq.NOISE_MODEL_LIKE'=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, split_untangled_states: bool=True):
        """Density matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            split_untangled_states: If True, optimizes simulation by running
                unentangled qubit sets independently and merging those states
                at the end.

        Raises:
            ValueError: If the supplied dtype is not `np.complex64` or
                `np.complex128`.

        Example:
           >>> (q0,) = cirq.LineQubit.range(1)
           >>> circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
        """
        super().__init__(dtype=dtype, noise=noise, seed=seed, split_untangled_states=split_untangled_states)
        if dtype not in {np.complex64, np.complex128}:
            raise ValueError(f'dtype must be complex64 or complex128, was {dtype}')

    def _create_partial_simulation_state(self, initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE', 'cirq.DensityMatrixSimulationState'], qubits: Sequence['cirq.Qid'], classical_data: 'cirq.ClassicalDataStore') -> 'cirq.DensityMatrixSimulationState':
        """Creates the DensityMatrixSimulationState for a circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            classical_data: The shared classical data container for this
                simulation.

        Returns:
            DensityMatrixSimulationState for the circuit.
        """
        if isinstance(initial_state, density_matrix_simulation_state.DensityMatrixSimulationState):
            return initial_state
        return density_matrix_simulation_state.DensityMatrixSimulationState(qubits=qubits, prng=self._prng, classical_data=classical_data, initial_state=initial_state, dtype=self._dtype)

    def _can_be_in_run_prefix(self, val: Any):
        return not protocols.measurement_keys_touched(val)

    def _create_step_result(self, sim_state: 'cirq.SimulationStateBase[cirq.DensityMatrixSimulationState]'):
        return DensityMatrixStepResult(sim_state=sim_state, dtype=self._dtype)

    def _create_simulator_trial_result(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[cirq.DensityMatrixSimulationState]') -> 'cirq.DensityMatrixTrialResult':
        return DensityMatrixTrialResult(params=params, measurements=measurements, final_simulator_state=final_simulator_state)

    def simulate_expectation_values_sweep(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> List[List[float]]:
        if not permit_terminal_measurements and program.are_any_measurements_terminal():
            raise ValueError('Provided circuit has terminal measurements, which may skew expectation values. If this is intentional, set permit_terminal_measurements=True.')
        swept_evs = []
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qmap = {q: i for i, q in enumerate(qubit_order.order_for(program.all_qubits()))}
        if not isinstance(observables, List):
            observables = [observables]
        pslist = [ops.PauliSum.wrap(pslike) for pslike in observables]
        for param_resolver in study.to_resolvers(params):
            result = self.simulate(program, param_resolver, qubit_order=qubit_order, initial_state=initial_state)
            swept_evs.append([obs.expectation_from_density_matrix(result.final_density_matrix, qmap) for obs in pslist])
        return swept_evs