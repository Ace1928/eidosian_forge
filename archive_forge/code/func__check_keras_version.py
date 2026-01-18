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
def _check_keras_version():
    from keras import __version__ as keras_version
    from wandb.util import parse_version
    if parse_version(keras_version) < parse_version('2.4.0'):
        wandb.termwarn(f'Keras version {keras_version} is not fully supported. Required keras >= 2.4.0')