import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def _normalize_hparams(hparams):
    """Normalize a dict keyed by `HParam`s and/or raw strings.

    Args:
      hparams: A `dict` whose keys are `HParam` objects and/or strings
        representing hyperparameter names, and whose values are
        hyperparameter values. No two keys may have the same name.

    Returns:
      A `dict` whose keys are hyperparameter names (as strings) and whose
      values are the corresponding hyperparameter values, after numpy
      normalization (see `_normalize_numpy_value`).

    Raises:
      ValueError: If two entries in `hparams` share the same
        hyperparameter name.
    """
    result = {}
    for k, v in hparams.items():
        if isinstance(k, HParam):
            k = k.name
        if k in result:
            raise ValueError('multiple values specified for hparam %r' % (k,))
        result[k] = _normalize_numpy_value(v)
    return result