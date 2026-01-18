import numpy as np
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def _check_shape(a, b):
    msg = f'Inferred static shapes are different between two loops: {a.shape} vs {b.shape}.'
    if b.shape:
        self.assertEqual(a.shape.as_list()[0], b.shape.as_list()[0], msg)