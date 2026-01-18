import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _raise_value_error_on_nested_artifact(self, v, nested=False):
    if isinstance(v, dict) and check_dict_contains_nested_artifact(v, nested):
        raise ValueError('Instances of wandb.Artifact can only be top level keys in wandb.config')