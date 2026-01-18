from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Metric as ProtoMetric
from mlflow.protos.service_pb2 import MetricWithRunId as ProtoMetricWithRunId
class MetricWithRunId(Metric):

    def __init__(self, metric: Metric, run_id):
        super().__init__(key=metric.key, value=metric.value, timestamp=metric.timestamp, step=metric.step)
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id

    def to_dict(self):
        return {'key': self.key, 'value': self.value, 'timestamp': self.timestamp, 'step': self.step, 'run_id': self.run_id}

    def to_proto(self):
        metric = ProtoMetricWithRunId()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        metric.step = self.step
        metric.run_id = self.run_id
        return metric