import collections
import copy
import functools
import itertools
import threading
import warnings
import weakref
import numpy as np
from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.tools.docs import doc_controls
def _init_call_fn_args(self, expects_training_arg=None):
    self.__class__._call_full_argspec.fget.cache.pop(self, None)
    self.__class__._call_fn_args.fget.cache.pop(self, None)
    self.__class__._call_accepts_kwargs.fget.cache.pop(self, None)
    call_fn_args = self._call_fn_args
    call_fn_args += self._call_full_argspec.kwonlyargs or []
    if expects_training_arg is None:
        self._expects_training_arg = 'training' in call_fn_args or self._call_accepts_kwargs
    else:
        self._expects_training_arg = expects_training_arg
    call_fn_arg_defaults = self._call_fn_arg_defaults.copy()
    call_fn_arg_defaults.update(self._call_full_argspec.kwonlydefaults or {})
    self._default_training_arg = call_fn_arg_defaults.get('training')
    self._expects_mask_arg = 'mask' in call_fn_args or self._call_accepts_kwargs