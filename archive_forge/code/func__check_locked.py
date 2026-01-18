import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _check_locked(self, key, ignore_locked=False) -> bool:
    locked = self._locked.get(key)
    if locked is not None:
        locked_user = self._users_inv[locked]
        if not ignore_locked:
            wandb.termwarn(f"Config item '{key}' was locked by '{locked_user}' (ignored update).")
        return True
    return False