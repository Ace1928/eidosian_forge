import inspect
import numpy as np
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
class MetricsTracker:
    """Record of the values of multiple executions of all metrics.

    It contains `MetricHistory` instances for the metrics.

    Args:
        metrics: List of strings of the names of the metrics.
    """

    def __init__(self, metrics=None):
        self.metrics = {}
        self.register_metrics(metrics)

    def exists(self, name):
        return name in self.metrics

    def register_metrics(self, metrics=None):
        metrics = metrics or []
        for metric in metrics:
            self.register(metric.name)

    def register(self, name, direction=None):
        if self.exists(name):
            raise ValueError(f'Metric already exists: {name}')
        if direction is None:
            direction = infer_metric_direction(name)
        if direction is None:
            direction = 'min'
        self.metrics[name] = MetricHistory(direction)

    def update(self, name, value, step=0):
        value = float(value)
        if not self.exists(name):
            self.register(name)
        prev_best = self.metrics[name].get_best_value()
        self.metrics[name].update(value, step=step)
        new_best = self.metrics[name].get_best_value()
        improved = new_best != prev_best
        return improved

    def get_history(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_history()

    def set_history(self, name, observations):
        if not self.exists(name):
            self.register(name)
        self.metrics[name].set_history(observations)

    def get_best_value(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_best_value()

    def get_best_step(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_best_step()

    def get_statistics(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_statistics()

    def get_last_value(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_last_value()

    def get_direction(self, name):
        self._assert_exists(name)
        return self.metrics[name].direction

    def get_config(self):
        return {'metrics': {name: metric_history.get_config() for name, metric_history in self.metrics.items()}}

    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.metrics = {name: MetricHistory.from_config(metric_history) for name, metric_history in config['metrics'].items()}
        return instance

    def to_proto(self):
        return protos.get_proto().MetricsTracker(metrics={name: metric_history.to_proto() for name, metric_history in self.metrics.items()})

    @classmethod
    def from_proto(cls, proto):
        instance = cls()
        instance.metrics = {name: MetricHistory.from_proto(metric_history) for name, metric_history in proto.metrics.items()}
        return instance

    def _assert_exists(self, name):
        if name not in self.metrics:
            raise ValueError(f'Unknown metric: {name}')