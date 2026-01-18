import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class GraphExecutionFunction:
    """Runs a computation graph.

  It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
  In particular additional operations via `fetches` argument and additional
  tensor substitutions via `feed_dict` arguments. Note that given
  substitutions are merged with substitutions from `inputs`. Even though
  `feed_dict` is passed once in the constructor (called in `model.compile()`)
  we can modify the values in the dictionary. Through this feed_dict we can
  provide additional substitutions besides Keras inputs.

  Args:
      inputs: Feed placeholders to the computation graph.
      outputs: Output tensors to fetch.
      updates: Additional update ops to be run at function call.
      name: A name to help users identify what this function does.
      session_kwargs: Arguments to `tf.Session.run()`:
                      `fetches`, `feed_dict`, `options`, `run_metadata`.
  """

    def __init__(self, inputs, outputs, updates=None, name=None, **session_kwargs):
        updates = updates or []
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a Keras backend function should be a list or tuple.')
        self._inputs_structure = inputs
        self.inputs = nest.flatten(inputs, expand_composites=True)
        self._outputs_structure = outputs
        self.outputs = cast_variables_to_tensor(nest.flatten(outputs, expand_composites=True))
        with ops.control_dependencies([self.outputs[0]]):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(state_ops.assign(p, new_p))
                else:
                    updates_ops.append(update)
            self.updates_op = control_flow_ops.group(*updates_ops)
        self.name = name
        self.feed_dict = session_kwargs.pop('feed_dict', None)
        self.fetches = session_kwargs.pop('fetches', [])
        if not isinstance(self.fetches, list):
            self.fetches = [self.fetches]
        self.run_options = session_kwargs.pop('options', None)
        self.run_metadata = session_kwargs.pop('run_metadata', None)
        self.fetches = [array_ops.identity(x) for x in self.fetches]
        self.session_kwargs = session_kwargs
        self.fetch_callbacks = {}
        if session_kwargs:
            raise ValueError('Some keys in session_kwargs are not supported at this time: %s' % (session_kwargs.keys(),))
        self._callable_fn = None
        self._feed_arrays = None
        self._feed_symbols = None
        self._symbol_vals = None
        self._fetches = None
        self._session = None

    def _make_callable(self, feed_arrays, feed_symbols, symbol_vals, session):
        """Generates a callable that runs the graph.

    Args:
      feed_arrays: List of input tensors to be fed Numpy arrays at runtime.
      feed_symbols: List of input tensors to be fed symbolic tensors at runtime.
      symbol_vals: List of symbolic tensors to be fed to `feed_symbols`.
      session: Session to use to generate the callable.

    Returns:
      Function that runs the graph according to the above options.
    """
        callable_opts = config_pb2.CallableOptions()
        for x in feed_arrays:
            callable_opts.feed.append(x.name)
        if self.feed_dict:
            for key in sorted(self.feed_dict.keys()):
                callable_opts.feed.append(key.name)
        for x, y in zip(feed_symbols, symbol_vals):
            connection = callable_opts.tensor_connection.add()
            if x.dtype != y.dtype:
                y = math_ops.cast(y, dtype=x.dtype)
            from_tensor = _as_graph_element(y)
            if from_tensor is None:
                from_tensor = y
            connection.from_tensor = from_tensor.name
            connection.to_tensor = x.name
        for x in self.outputs + self.fetches:
            callable_opts.fetch.append(x.name)
        callable_opts.target.append(self.updates_op.name)
        if self.run_options:
            callable_opts.run_options.CopyFrom(self.run_options)
        callable_fn = session._make_callable_from_options(callable_opts)
        self._callable_fn = callable_fn
        self._feed_arrays = feed_arrays
        self._feed_symbols = feed_symbols
        self._symbol_vals = symbol_vals
        self._fetches = list(self.fetches)
        self._session = session

    def _call_fetch_callbacks(self, fetches_output):
        for fetch, output in zip(self._fetches, fetches_output):
            if fetch in self.fetch_callbacks:
                self.fetch_callbacks[fetch](output)

    def _eval_if_composite(self, tensor):
        """Helper method which evaluates any CompositeTensors passed to it."""
        from tensorflow.python.keras.utils import tf_utils
        if tf_utils.is_extension_type(tensor):
            return self._session.run(tensor)
        else:
            return tensor

    def __call__(self, inputs):
        inputs = nest.flatten(inputs, expand_composites=True)
        session = get_session(inputs)
        feed_arrays = []
        array_vals = []
        feed_symbols = []
        symbol_vals = []
        for tensor, value in zip(self.inputs, inputs):
            if value is None:
                continue
            if tensor_util.is_tf_type(value):
                feed_symbols.append(tensor)
                symbol_vals.append(value)
            else:
                feed_arrays.append(tensor)
                tensor_type = dtypes_module.as_dtype(tensor.dtype)
                array_vals.append(np.asarray(value, dtype=tensor_type.as_numpy_dtype))
        if self.feed_dict:
            for key in sorted(self.feed_dict.keys()):
                array_vals.append(np.asarray(self.feed_dict[key], dtype=key.dtype.as_numpy_dtype))
        if self._callable_fn is None or feed_arrays != self._feed_arrays or symbol_vals != self._symbol_vals or (feed_symbols != self._feed_symbols) or (self.fetches != self._fetches) or (session != self._session):
            self._make_callable(feed_arrays, feed_symbols, symbol_vals, session)
        fetched = self._callable_fn(*array_vals, run_metadata=self.run_metadata)
        self._call_fetch_callbacks(fetched[-len(self._fetches):])
        output_structure = nest.pack_sequence_as(self._outputs_structure, fetched[:len(self.outputs)], expand_composites=True)
        return nest.map_structure(self._eval_if_composite, output_structure)