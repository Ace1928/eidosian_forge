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
@contextmanager
def _get_checkpoint(self, model: Booster, epoch: int, filename: str, frequency: int) -> Optional[Checkpoint]:
    if not frequency or epoch % frequency > 0 or (not epoch and frequency > 1):
        yield None
        return
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_model(os.path.join(checkpoint_dir, filename))
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        yield checkpoint