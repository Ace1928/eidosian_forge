import logging
import os
import tempfile
import warnings
from packaging.version import Version
import mlflow.pytorch
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.pytorch import _pytorch_autolog
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
class __MlflowPLCallback(pl.Callback, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for auto-logging metrics and parameters.
    """

    def __init__(self, client, metrics_logger, run_id, log_models, log_every_n_epoch, log_every_n_step):
        if log_every_n_step and _pl_version < Version('1.1.0'):
            raise MlflowException('log_every_n_step is only supported for PyTorch-Lightning >= 1.1.0')
        self.early_stopping = False
        self.client = client
        self.metrics_logger = metrics_logger
        self.run_id = run_id
        self.log_models = log_models
        self.log_every_n_epoch = log_every_n_epoch
        self.log_every_n_step = log_every_n_step
        self._global_steps_per_training_step = 1
        self._step_metrics = set()
        self._epoch_metrics = set()

    def _log_metrics(self, trainer, step, metric_items):
        sanity_checking = trainer.sanity_checking if Version(pl.__version__) > Version('1.4.5') else trainer.running_sanity_check
        if sanity_checking:
            return
        metrics = {x[0]: float(x[1]) for x in metric_items}
        self.metrics_logger.record_metrics(metrics, step)

    def _log_epoch_metrics(self, trainer, pl_module):
        metric_items = [(name, val) for name, val in trainer.callback_metrics.items() if name not in self._step_metrics]
        self._epoch_metrics.update((name for name, _ in metric_items))
        if (pl_module.current_epoch + 1) % self.log_every_n_epoch == 0:
            self._log_metrics(trainer, pl_module.current_epoch, metric_items)
    _pl_version = Version(pl.__version__)
    if _pl_version >= Version('1.4.0dev'):

        @rank_zero_only
        def on_train_epoch_end(self, trainer, pl_module, *args):
            self._log_epoch_metrics(trainer, pl_module)
    elif _pl_version >= Version('1.2.0'):

        @rank_zero_only
        def on_train_epoch_end(self, trainer, pl_module, *args):
            """
            Log loss and other metrics values after each train epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            """
            if not trainer.enable_validation:
                self._log_epoch_metrics(trainer, pl_module)

        @rank_zero_only
        def on_validation_epoch_end(self, trainer, pl_module):
            """
            Log loss and other metrics values after each validation epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            """
            self._log_epoch_metrics(trainer, pl_module)
    else:

        @rank_zero_only
        def on_epoch_end(self, trainer, pl_module):
            """
            Log loss and other metrics values after each epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            """
            self._log_epoch_metrics(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args):
        """
        Log metric values after each step

        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
        """
        if not self.log_every_n_step:
            return
        metrics = _get_step_metrics(trainer)
        metric_items = [(name, val) for name, val in metrics.items() if name not in self._epoch_metrics and f'{name}_step' not in metrics.keys()]
        self._step_metrics.update((name for name, _ in metric_items))
        step = trainer.global_step
        if (step // self._global_steps_per_training_step + 1) % self.log_every_n_step == 0:
            self._log_metrics(trainer, step, metric_items)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """
        Logs Optimizer related metrics when the train begins

        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
        """
        self.client.set_tags(self.run_id, {'Mode': 'training'})
        params = {'epochs': trainer.max_epochs}
        if hasattr(trainer, 'optimizers'):
            if _pl_version >= Version('1.6.0'):
                self._global_steps_per_training_step = len(trainer.optimizers)
            optimizer = trainer.optimizers[0]
            params['optimizer_name'] = _get_optimizer_name(optimizer)
            if hasattr(optimizer, 'defaults'):
                params.update(optimizer.defaults)
        self.client.log_params(self.run_id, params)
        self.client.flush(synchronous=True)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """
        Logs the model checkpoint into mlflow - models folder on the training end


        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
        """
        self.metrics_logger.flush()
        self.client.flush(synchronous=True)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        """
        Logs accuracy and other relevant metrics on the testing end

        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
        """
        self.client.set_tags(self.run_id, {'Mode': 'testing'})
        self.client.flush(synchronous=True)
        self.metrics_logger.record_metrics({key: float(value) for key, value in trainer.callback_metrics.items()})
        self.metrics_logger.flush()