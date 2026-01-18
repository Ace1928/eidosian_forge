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
def _create_seed(self, user_specified_seed):
    if user_specified_seed is not None:
        return user_specified_seed
    elif getattr(_SEED_GENERATOR, 'generator', None):
        return _SEED_GENERATOR.generator.randint(1, 1000000000.0)
    else:
        return random.randint(1, int(1000000000.0))