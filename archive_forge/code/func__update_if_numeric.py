import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def _update_if_numeric(metrics, key, values):
    if not _array_has_dtype(values):
        _warn_not_logging(key)
        return
    if not is_numeric_array(values):
        _warn_not_logging_non_numeric(key)
        return
    metrics[key] = wandb.Histogram(values)