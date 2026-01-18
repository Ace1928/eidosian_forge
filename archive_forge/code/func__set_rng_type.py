import collections
import itertools
import json
import os
import random
import sys
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.distribute import distribute_coordinator_utils as dc
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.engine import keras_tensor
from keras.src.utils import control_flow_util
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _set_rng_type(self, rng_type, **kwargs):
    if kwargs.get('force_generator', False):
        rng_type = self.RNG_STATEFUL
    if rng_type is None:
        if is_tf_random_generator_enabled():
            self._rng_type = self.RNG_STATEFUL
        else:
            self._rng_type = self.RNG_LEGACY_STATEFUL
    else:
        if rng_type not in [self.RNG_STATEFUL, self.RNG_LEGACY_STATEFUL, self.RNG_STATELESS]:
            raise ValueError(f'Invalid `rng_type` received. Valid `rng_type` are ["stateless", "stateful", "legacy_stateful"]. Got: {rng_type}')
        self._rng_type = rng_type