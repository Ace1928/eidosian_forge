import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) + self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(x_f + backend.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) + self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(x_c + backend.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) + self.output_gate_peephole_weights * c)
    return (c, o)