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
class EstimatorSpecFunction(tf.compat.v2.__internal__.function.Function):
    """Wraps graph functions defined for a function returning an EstimatorSpec.

  Instances of this class are revivable when attached to a checkpointable
  object.
  """

    def __init__(self, fn, mode, config=None, params=None, variable_holder=None, **kwargs):
        """Initializes an EstimatorSpecFunction.

    Args:
      fn: Python model function.
      mode: String mode to run the function.
      config: RunConfig that is passed to the `config` arg in the function.
      params: object that is passed to the `params` argument in the function.
      variable_holder: Optional `wrap_function.VariableHolder` object.
      **kwargs: Optional keyword arguments to pass to tf.function (e.g.
        input_signature).
    """
        python_function, self.expects_labels = _wrap_and_verify_model_fn(fn, mode=mode, config=config, params=params, input_signature=kwargs.get('input_signature', None))
        super(EstimatorSpecFunction, self).__init__(python_function, mode, **kwargs)
        self._variable_holder = variable_holder

    def _defun(self, fn):
        return _EstimatorSpecFunction(fn, name=self._name, variable_holder=self._variable_holder, input_signature=self.input_signature, autograph=self._autograph, autograph_options=self._experimental_autograph_options)