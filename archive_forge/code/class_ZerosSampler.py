import abc
from typing import List, Optional, TYPE_CHECKING
import numpy as np
from cirq import devices, work, study
class ZerosSampler(work.Sampler, metaclass=abc.ABCMeta):
    """A mock sampler for testing. Immediately returns zeroes."""

    def __init__(self, device: Optional[devices.Device]=None):
        """Construct a sampler that returns 0 for all measurements.

        Args:
            device: A device against which to validate the circuit. If None,
                no validation will be done.
        """
        self.device = device

    def run_sweep(self, program: 'cirq.AbstractCircuit', params: study.Sweepable, repetitions: int=1) -> List[study.Result]:
        """Samples circuit as if every measurement resulted in zero.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.

        Raises:
            ValueError: circuit is not valid for the sampler, due to invalid
            repeated keys or incompatibility with the sampler's device.
        """
        if self.device:
            self.device.validate_circuit(program)
        shapes = self._get_measurement_shapes(program)
        return [study.ResultDict(params=param_resolver, records={k: np.zeros((repetitions, num_instances, len(qid_shape)), dtype=int) for k, (num_instances, qid_shape) in shapes.items()}) for param_resolver in study.to_resolvers(params)]