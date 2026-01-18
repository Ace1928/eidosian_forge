from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import func_graph
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class ModelFunction(tf.compat.v2.__internal__.tracking.AutoTrackable):
    """A checkpointable ModelFunction object.

  This object stores a global mapping of variables and functions for each mode.
  """

    def __init__(self, config=None, params=None):
        self._config = config
        self._params = params
        self._functions = {}
        self._variable_holder = wrap_function.VariableHolder(share_variables=True)
        self._variables_by_name = self._variable_holder.variables

    @staticmethod
    def from_function(model_fn, all_modes=None, config=None, params=None):
        """Creates a new ModelFunction object from a model function."""
        if all_modes is None:
            all_modes = [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]
        else:
            all_modes = list(all_modes)
        obj = ModelFunction(config=config, params=params)
        for mode in all_modes:
            obj.add_mode(model_fn, mode)
        return obj

    @property
    def variables(self):
        return self._variables_by_name

    def add_mode(self, fn, mode, input_signature=None):
        if mode in self._functions:
            raise ValueError('ModelFunction object has multiple functions with name {}.'.format(mode))
        spec_fn = EstimatorSpecFunction(fn, mode, config=self._config, params=self._params, variable_holder=self._variable_holder, input_signature=input_signature)
        self._functions[mode] = spec_fn

    def train(self, features, labels):
        return self.call(ModeKeys.TRAIN, features, labels)

    def evaluate(self, features, labels):
        return self.call(ModeKeys.EVAL, features, labels)

    def predict(self, features):
        return self.call(ModeKeys.PREDICT, features)

    def call(self, mode, features, labels=None):
        if mode not in self._functions:
            raise ValueError('Mode {} is not defined the ModelFunction. To add modes, use the `add_mode()` function. Available modes: {}'.format(mode, self._functions.keys()))
        fn = self._functions[mode]
        if fn.expects_labels:
            return fn(features, labels)
        else:
            return fn(features)