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
def _step(time, output_ta_t, *states):
    """RNN step function.

        Args:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
    current_input = tuple((ta.read(time) for ta in input_ta))
    current_input = nest.pack_sequence_as(inputs, current_input)
    output, new_states = step_function(current_input, tuple(states) + tuple(constants))
    flat_state = nest.flatten(states)
    flat_new_state = nest.flatten(new_states)
    for state, new_state in zip(flat_state, flat_new_state):
        if isinstance(new_state, tensor_lib.Tensor):
            new_state.set_shape(state.shape)
    flat_output = nest.flatten(output)
    output_ta_t = tuple((ta.write(time, out) for ta, out in zip(output_ta_t, flat_output)))
    new_states = nest.pack_sequence_as(initial_states, flat_new_state)
    return (time + 1, output_ta_t) + tuple(new_states)