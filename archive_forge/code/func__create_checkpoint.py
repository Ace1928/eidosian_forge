from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from lightgbm.callback import CallbackEnv
from lightgbm.basic import Booster
from ray.util.annotations import Deprecated
@staticmethod
def _create_checkpoint(model: Booster, epoch: int, filename: str, frequency: int):
    if log_once('lightgbm_ray_legacy'):
        warnings.warn("You are using an outdated version of LightGBM-Ray that won't be compatible with future releases of Ray. Please update LightGBM-Ray with `pip install -U lightgbm_ray`.")
    if not frequency or epoch % frequency > 0 or (not epoch and frequency > 1):
        return
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        model.save_model(os.path.join(checkpoint_dir, filename))