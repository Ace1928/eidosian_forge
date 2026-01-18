import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _filter_ds(dataset, acceptance_dist_ds, initial_dist_ds, class_func, seed, name=None):
    """Filters a dataset based on per-class acceptance probabilities.

  Args:
    dataset: The dataset to be filtered.
    acceptance_dist_ds: A dataset of acceptance probabilities.
    initial_dist_ds: A dataset of the initial probability distribution, given or
      estimated.
    class_func: A function mapping an element of the input dataset to a scalar
      `tf.int32` tensor. Values should be in `[0, num_classes)`.
    seed: (Optional.) Python integer seed for the resampler.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A dataset of (class value, data) after filtering.
  """

    def maybe_warn_on_large_rejection(accept_dist, initial_dist):
        proportion_rejected = math_ops.reduce_sum((1 - accept_dist) * initial_dist)
        return cond.cond(math_ops.less(proportion_rejected, 0.5), lambda: accept_dist, lambda: logging_ops.Print(accept_dist, [proportion_rejected, initial_dist, accept_dist], message='Proportion of examples rejected by sampler is high: ', summarize=100, first_n=10))
    acceptance_dist_ds = DatasetV2.zip((acceptance_dist_ds, initial_dist_ds), name=name).map(maybe_warn_on_large_rejection, name=name)

    def _gather_and_copy(acceptance_prob, data):
        if isinstance(data, tuple):
            class_val = class_func(*data)
        else:
            class_val = class_func(data)
        return (class_val, array_ops.gather(acceptance_prob, class_val), data)
    current_probabilities_and_class_and_data_ds = DatasetV2.zip((acceptance_dist_ds, dataset), name=name).map(_gather_and_copy, name=name)

    def _reject(unused_class_val, p, unused_data):
        return random_ops.random_uniform([], seed=seed, dtype=p.dtype) < p
    filtered_ds = current_probabilities_and_class_and_data_ds.filter(_reject, name=name)
    return filtered_ds.map(lambda class_value, _, data: (class_value, data), name=name)