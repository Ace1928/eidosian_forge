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
def _get_optimizer_name(optimizer):
    """
    In pytorch-lightining 1.1.0, `LightningOptimizer` was introduced:
    https://github.com/PyTorchLightning/pytorch-lightning/pull/4658

    If a user sets `enable_pl_optimizer` to True when instantiating a `Trainer` object,
    each optimizer will be wrapped by `LightningOptimizer`:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html
    #pytorch_lightning.trainer.trainer.Trainer.params.enable_pl_optimizer
    """
    if Version(pl.__version__) < Version('1.1.0'):
        return optimizer.__class__.__name__
    else:
        from pytorch_lightning.core.optimizer import LightningOptimizer
        return optimizer._optimizer.__class__.__name__ if isinstance(optimizer, LightningOptimizer) else optimizer.__class__.__name__