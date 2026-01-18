import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
class SimulatesSamples(work.Sampler, metaclass=abc.ABCMeta):
    """Simulator that mimics running on quantum hardware.

    Implementors of this interface should implement the _run method.
    """

    def run_sweep(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int=1) -> Sequence['cirq.Result']:
        return list(self.run_sweep_iter(program, params, repetitions))

    def run_sweep_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int=1) -> Iterator['cirq.Result']:
        """Runs the supplied Circuit, mimicking quantum hardware.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.

        Raises:
            ValueError: If the circuit has no measurements.
        """
        if not program.has_measurements():
            raise ValueError('Circuit has no measurements to sample.')
        for param_resolver in study.to_resolvers(params):
            records = {}
            if repetitions == 0:
                for _, op, _ in program.findall_operations_with_gate_type(ops.MeasurementGate):
                    records[protocols.measurement_key_name(op)] = np.empty([0, 1, 1])
            else:
                records = self._run(circuit=program, param_resolver=param_resolver, repetitions=repetitions)
            yield study.ResultDict(params=param_resolver, records=records)

    @abc.abstractmethod
    def _run(self, circuit: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolver', repetitions: int) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run. It is expected that
                this is validated greater than zero before calling this method.

        Returns:
            A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 3-dimensional
            numpy array, the first dimension corresponding to the repetition.
            the second to the instance of that key in the circuit, and the
            third to the actual boolean measurement results (ordered by the
            qubits being measured.)
        """
        raise NotImplementedError()