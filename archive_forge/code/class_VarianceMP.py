import warnings
from typing import Sequence, Tuple, Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import SampleMeasurement, StateMeasurement, Variance
from .mid_measure import MeasurementValue
class VarianceMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the variance of the supplied observable.

    Please refer to :func:`var` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    @property
    def return_type(self):
        return Variance

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum((s.copies for s in shots.shot_vector))
        return tuple((() for _ in range(num_shot_elements)))

    def process_samples(self, samples: Sequence[complex], wire_order: Wires, shot_range: Tuple[int]=None, bin_size: int=None):
        op = self.mv if self.mv is not None else self.obs
        with qml.queuing.QueuingManager.stop_recording():
            samples = qml.sample(op=op).process_samples(samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size)
        axis = -1 if bin_size is None else -2
        return qml.math.squeeze(qml.math.var(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        eigvals = qml.math.asarray(self.eigvals(), dtype='float64')
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        return qml.math.dot(prob, eigvals ** 2) - qml.math.dot(prob, eigvals) ** 2