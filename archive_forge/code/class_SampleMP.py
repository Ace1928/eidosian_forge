import functools
import warnings
from typing import Sequence, Tuple, Optional, Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import MeasurementShapeError, Sample, SampleMeasurement
from .mid_measure import MeasurementValue
class SampleMP(SampleMeasurement):
    """Measurement process that returns the samples of a given observable. If no observable is
    provided then basis state samples are returned directly from the device.

    Please refer to :func:`sample` for detailed documentation.

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
        return Sample

    @property
    @functools.lru_cache()
    def numeric_type(self):
        if self.obs is None:
            return int
        int_eigval_obs = {qml.X, qml.Y, qml.Z, qml.Hadamard, qml.Identity}
        tensor_terms = self.obs.obs if hasattr(self.obs, 'obs') else [self.obs]
        every_term_standard = all((o.__class__ in int_eigval_obs for o in tensor_terms))
        return int if every_term_standard else float

    def shape(self, device, shots):
        if not shots:
            raise MeasurementShapeError(f'Shots are required to obtain the shape of the measurement {self.__class__.__name__}.')
        len_wires = len(self.wires) if len(self.wires) > 0 else len(device.wires)

        def _single_int_shape(shot_val, num_wires):
            inner_shape = []
            if shot_val != 1:
                inner_shape.append(shot_val)
            if num_wires != 1:
                inner_shape.append(num_wires)
            return tuple(inner_shape)
        if not shots.has_partitioned_shots:
            return _single_int_shape(shots.total_shots, len_wires)
        shape = []
        for s in shots.shot_vector:
            for _ in range(s.copies):
                shape.append(_single_int_shape(s.shots, len_wires))
        return tuple(shape)

    def process_samples(self, samples: Sequence[complex], wire_order: Wires, shot_range: Tuple[int]=None, bin_size: int=None):
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        name = self.obs.name if self.obs is not None else None
        if shot_range is not None:
            samples = samples[..., slice(*shot_range), :]
        if mapped_wires:
            samples = samples[..., mapped_wires]
        num_wires = samples.shape[-1]
        if self.obs is None and (not isinstance(self.mv, MeasurementValue)):
            return samples if bin_size is None else samples.T.reshape(num_wires, bin_size, -1)
        if str(name) in {'PauliX', 'PauliY', 'PauliZ', 'Hadamard'}:
            samples = 1 - 2 * qml.math.squeeze(samples, axis=-1)
        else:
            powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]
            indices = samples @ powers_of_two
            indices = qml.math.array(indices)
            try:
                samples = self.eigvals()[indices]
            except qml.operation.EigvalsUndefinedError as e:
                raise qml.operation.EigvalsUndefinedError(f'Cannot compute samples of {self.obs.name}.') from e
        return samples if bin_size is None else samples.reshape((bin_size, -1))