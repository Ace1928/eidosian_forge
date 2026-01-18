import collections.abc as collections
import json
import os
import warnings
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.utils import (
from .constants import CONFIG_NAME
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
def _create_hyperparameter_table(model):
    """Parse hyperparameter dictionary into a markdown table."""
    if model.optimizer is not None:
        optimizer_params = model.optimizer.get_config()
        optimizer_params = _flatten_dict(optimizer_params)
        optimizer_params['training_precision'] = tf.keras.mixed_precision.global_policy().name
        table = '| Hyperparameters | Value |\n| :-- | :-- |\n'
        for key, value in optimizer_params.items():
            table += f'| {key} | {value} |\n'
    else:
        table = None
    return table