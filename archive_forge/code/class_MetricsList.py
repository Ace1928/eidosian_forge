import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
class MetricsList(metrics_module.Metric):

    def __init__(self, metrics, name='metrics_list', output_name=None):
        super().__init__(name=name)
        self.metrics = metrics
        self.output_name = output_name

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.metrics:
            m.update_state(y_true, y_pred, sample_weight=sample_weight)

    def reset_state(self):
        for m in self.metrics:
            m.reset_state()

    def get_result(self):
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError