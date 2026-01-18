import numpy as np
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def _run_targets(self, targets1, targets2=None, run_init=True):
    targets1 = nest.flatten(targets1)
    targets2 = [] if targets2 is None else nest.flatten(targets2)
    assert len(targets1) == len(targets2) or not targets2
    if run_init:
        init = variables.global_variables_initializer()
        self.evaluate(init)
    return self.evaluate(targets1 + targets2)