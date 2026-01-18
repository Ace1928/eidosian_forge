from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
class _InputsWithStoppingSignals(_Inputs):
    """Inputs with `_StopSignals` inserted into the dataset."""

    def __init__(self, dataset, batch_size, add_padding=False, num_invocations_per_step=1):
        assert dataset is not None
        user_provided_dataset = dataset.map(_InputsWithStoppingSignals.insert_stopping_signal(stop=False, batch_size=batch_size, add_padding=add_padding))
        if num_invocations_per_step == 1:
            final_batch_dataset = dataset.take(1).map(_InputsWithStoppingSignals.insert_stopping_signal(stop=True, batch_size=batch_size, add_padding=add_padding))
        else:
            final_batch_dataset = dataset.take(1).map(_InputsWithStoppingSignals.insert_stopping_signal(stop=True, batch_size=batch_size, add_padding=add_padding))
            final_batch_dataset = final_batch_dataset.repeat(2 * num_invocations_per_step - 1)

            def _set_mask(data_dict):
                signals = data_dict['signals']
                signals['padding_mask'] = tf.compat.v1.ones_like(signals['padding_mask'])
                data_dict['signals'] = signals
                return data_dict
            final_batch_dataset = final_batch_dataset.map(_set_mask)
        dataset = user_provided_dataset.concatenate(final_batch_dataset).prefetch(2)
        super(_InputsWithStoppingSignals, self).__init__(dataset=dataset)
        self._current_inputs = None

    def features_and_labels(self):
        if self._current_inputs is not None:
            raise RuntimeError('Internal Error: The previous inputs have not been properly consumed. First call features_and_labels, then call signals.')
        inputs_with_signals = self._iterator.get_next()
        features = inputs_with_signals['features']
        labels = inputs_with_signals.get('labels')
        self._current_inputs = inputs_with_signals
        return (features, labels)

    def signals(self):
        """Returns the `Signals` from `_Inputs`."""
        if self._current_inputs is None:
            raise RuntimeError('Internal Error: The current inputs have not been properly generated. First call features_and_labels, then call signals.')
        signals = self._current_inputs['signals']
        self._current_inputs = None
        return signals

    @staticmethod
    def insert_stopping_signal(stop, batch_size, add_padding=False):
        """Inserts stopping_signal into dataset via _map_fn.

    Here we change the data structure in the dataset, such that the return value
    is a dictionary now and `features`, `labels`, and `signals` are three
    distinguished keys in that dict. This provides a better structure, which
    eases the process to decompose the inputs (see `features_and_labels`).

    Args:
      stop: bool, state of current stopping signals.
      batch_size: int, batch size.
      add_padding: bool, whether to pad the tensor to full batch size.

    Returns:
      A map_fn passed to dataset.map API.
    """

        def _map_fn(*args):
            """The map fn to insert signals."""
            if len(args) == 1:
                args = args[0]
            features, labels = _Inputs._parse_inputs(args)
            new_input_dict = {}
            if add_padding:
                padding_mask, features, labels = _PaddingSignals.pad_features_and_labels(features, labels, batch_size)
                new_input_dict['features'] = features
                if labels is not None:
                    new_input_dict['labels'] = labels
            else:
                new_input_dict['features'] = features
                if labels is not None:
                    new_input_dict['labels'] = labels
                padding_mask = None
            new_input_dict['signals'] = _StopSignals(stop=stop, batch_size=batch_size, padding_mask=padding_mask).as_dict()
            return new_input_dict
        return _map_fn