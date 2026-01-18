import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _set_artifact_callback(self, cb):
    object.__setattr__(self, '_artifact_callback', cb)