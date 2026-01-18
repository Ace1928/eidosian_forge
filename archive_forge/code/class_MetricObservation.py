import inspect
import numpy as np
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
class MetricObservation:
    """Metric value at a given step of training across multiple executions.

    If the model is trained multiple
    times (multiple executions), KerasTuner records the value of each
    metric at each training step. These values are aggregated
    over multiple executions into a list where each value corresponds
    to one execution.

    Args:
        value: Float or a list of floats. The evaluated metric values.
        step: Int. The step of the evaluation, for example, the epoch number.
    """

    def __init__(self, value, step):
        if not isinstance(value, list):
            value = [value]
        self.value = value
        self.step = step

    def append(self, value):
        if not isinstance(value, list):
            value = [value]
        self.value += value

    def mean(self):
        return np.mean(self.value)

    def get_config(self):
        return {'value': self.value, 'step': self.step}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __eq__(self, other):
        return other.value == self.value and other.step == self.step if isinstance(other, MetricObservation) else False

    def __repr__(self):
        return f'MetricObservation(value={self.value}, step={self.step})'

    def to_proto(self):
        return protos.get_proto().MetricObservation(value=self.value, step=self.step)

    @classmethod
    def from_proto(cls, proto):
        return cls(value=list(proto.value), step=proto.step)