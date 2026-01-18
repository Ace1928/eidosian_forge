import paddle
import mlflow
from mlflow.utils.autologging_utils import (
class __MlflowPaddleCallback(paddle.callbacks.Callback, metaclass=ExceptionSafeAbstractClass):
    """Callback for auto-logging metrics and parameters."""

    def __init__(self, client, metrics_logger, run_id, log_models, log_every_n_epoch):
        super().__init__()
        self.early_stopping = False
        self.client = client
        self.metrics_logger = metrics_logger
        self.run_id = run_id
        self.log_models = log_models
        self.log_every_n_epoch = log_every_n_epoch
        self.epoch = 0

    def _log_metrics(self, logs, current_epoch):
        metrics = {key: metric[0] if isinstance(metric, list) else metric for key, metric in logs.items()}
        self.metrics_logger.record_metrics(metrics, current_epoch)

    def on_epoch_end(self, epoch, logs=None):
        if self.model is not None and epoch % self.log_every_n_epoch == 0:
            self._log_metrics(logs, epoch)
            self.epoch = epoch

    def on_train_begin(self, logs=None):
        params = {'optimizer_name': self.model._optimizer.__class__.__name__, 'learning_rate': self.model._optimizer._learning_rate}
        self.client.log_params(self.run_id, params)
        self.client.flush(synchronous=True)

    def on_train_end(self, logs=None):
        self.metrics_logger.flush()
        self.client.flush(synchronous=True)

    def on_eval_end(self, logs=None):
        eval_logs = {'eval_' + key: metric[0] if isinstance(metric, list) else metric for key, metric in logs.items()}
        self._log_metrics(eval_logs, self.epoch)