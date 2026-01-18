from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
class _InitializeClustersHook(tf.compat.v1.train.SessionRunHook):
    """Initializes the cluster centers.

  The chief repeatedly invokes an initialization op until all cluster centers
  are initialized. The workers wait for the initialization phase to complete.
  """

    def __init__(self, init_op, is_initialized_var, is_chief):
        """Creates an _InitializeClustersHook.

    Args:
      init_op: An op that, when run, will choose some initial cluster centers.
        This op may need to be run multiple times to choose all the centers.
      is_initialized_var: A boolean variable reporting whether all initial
        centers have been chosen.
      is_chief: A boolean specifying whether this task is the chief.
    """
        self._init_op = init_op
        self._is_initialized_var = is_initialized_var
        self._is_chief = is_chief

    def after_create_session(self, session, coord):
        del coord
        assert self._init_op.graph is tf.compat.v1.get_default_graph()
        assert self._is_initialized_var.graph is self._init_op.graph
        while True:
            try:
                if session.run(self._is_initialized_var):
                    break
                elif self._is_chief:
                    session.run(self._init_op)
                else:
                    time.sleep(1)
            except RuntimeError as e:
                tf.compat.v1.logging.info(e)