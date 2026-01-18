import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
class VariableScopeWrapperLayer(base_layer.Layer):
    """Wrapper Layer to capture `compat.v1.get_variable` and `compat.v1.layers`.

  See go/tf2-migration-model-bookkeeping for background.

  This shim layer allows using large sets of TF1 model-forward-pass code as a
  Keras layer that works in TF2 with TF2 behaviors enabled. To use it,
  override this class and put your TF1 model's forward pass inside your
  implementation for `forward_pass`.

  Below are some examples, and then more details on the functionality of this
  shhim layer to wrap TF1 model forward passes.

  Example of capturing tf.compat.v1.layer-based modeling code as a Keras layer:

  ```python
  class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeWrapperLayer):

    def __init__(self, units, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.units = units

    def forward_pass(self, inputs, training=None):
      out = tf.compat.v1.layers.dense(
          inputs, self.units, name="dense_one",
          kernel_initializer=init_ops.ones_initializer(),
          kernel_regularizer="l2")
      with variable_scope.variable_scope("nested_scope"):
        out = tf.compat.v1.layers.dense(
            out, self.units, name="dense_two",
            kernel_initializer=init_ops.ones_initializer(),
            kernel_regularizer="l2")
      return out

  # Create a layer that can be used as a standard keras layer
  layer = WrappedDoubleDenseLayer(10)

  # call the layer on inputs
  layer(...)

  # Variables created/used within the scope will be tracked by the layer
  layer.weights
  layer.trainable_variables

  # Regularization losses will be captured in layer.losses after a call,
  # just like any other Keras layer
  reg_losses = layer.losses
  ```

  The solution is to wrap the model construction and execution in a keras-style
  scope:

  ```python
  class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeWrapperLayer):

    def __init__(self, units, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.units = units

    def forward_pass(self, inputs, training=None):
      out = inputs
      with tf.compat.v1.variable_scope("dense_one"):
        # The weights are created with a `regularizer`,
        # so the layer should track their regularization losses
        kernel = tf.compat.v1.get_variable(
            shape=[out.shape[-1], self.units],
            regularizer=regularizers.L2(),
            initializer=init_ops.ones_initializer(),
            name="kernel")
        bias = tf.compat.v1.get_variable(
            shape=[self.units,],
            initializer=init_ops.zeros_initializer(),
            name="bias")
        out = tf.compat.v1.math.matmul(out, kernel)
        out = tf.compat.v1.nn.bias_add(out, bias)
      with tf.compat.v1.variable_scope("nested_scope"):
        with tf.compat.v1.variable_scope("dense_two"):
          kernel = tf.compat.v1.get_variable(
              shape=[out.shape[-1], self.units],
              regularizer=regularizers.L2(),
              initializer=init_ops.ones_initializer(),
              name="kernel")
          bias = tf.compat.v1.get_variable(
              shape=[self.units,],
              initializer=init_ops.zeros_initializer(),
              name="bias")
          out = tf.compat.v1.math.matmul(out, kernel)
          out = tf.compat.v1.nn.bias_add(out, bias)
      return out

  # Create a layer that can be used as a standard keras layer
  layer = WrappedDoubleDenseLayer(10)

  # call the layer on inputs
  layer(...)

  # Variables created/used within the scope will be tracked by the layer
  layer.weights
  layer.trainable_variables

  # Regularization losses will be captured in layer.losses after a call,
  # just like any other Keras layer
  reg_losses = layer.losses
  ```

  Regularization losses:
    Any regularizers specified in the `get_variable` calls or `compat.v1.layer`
    creations will get captured by this wrapper layer. Regularization losses
    are accessible in `layer.losses` after a call just like in a standard
    Keras layer, and will be captured by any model that includes this layer.

  Variable scope / variable reuse:
    variable-scope based reuse in the `forward_pass` will be respected,
    and work like variable-scope based reuse in TF1.

  Variable Names/Pre-trained checkpoint loading:
    variable naming from get_variable and `compat.v1.layer` layers will match
    the TF1 names, so you should be able to re-use your old name-based
    checkpoints.

  Training Arg in `forward_pass`:
    Keras will pass a `training` arg to this layer similarly to how it
    passes `training` to other layers in TF2. See more details in the docs
    on `tf.keras.layers.Layer` to understand what will be passed and when.
    Note: tf.compat.v1.layers are usually not called with `training=None`,
    so the training arg to `forward_pass` might not feed through to them
    unless you pass it to their calls explicitly.

  Call signature of the forward pass:
    The semantics of the forward pass signature roughly match the standard
    Keras layer `call` signature, except that a `training` arg will *always*
    be passed, so your `forward_pass` must accept either.

  Limitations:
    * TF2 will not prune unused variable updates (or unused outputs). You may
      need to adjust your forward pass code to avoid computations or variable
      updates that you don't intend to use. (E.g. by adding a flag to the
      `forward_pass` call signature and branching on it).
    * Avoid Nesting variable creation in tf.function inside of `forward_pass`
      While the layer may safetely be used from inside a `tf.function`, using
      a function inside of `forward_pass` will break the variable scoping.
    * TBD: Nesting keras layers/models or other `VariableScopeWrapperLayer`s
      directly in `forward_pass` may not work correctly just yet.
      Support for this/instructions for how to do this is sill being worked on.

  Coming soon: A better guide, testing/verification guide.
  """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker = VariableAndLossTracker()

    def forward_pass(self, *args, **kwargs):
        raise NotImplementedError

    def call(self, *args, **kwargs):
        with self.tracker.scope():
            out = self.forward_pass(*args, **kwargs)
        if not self._eager_losses:
            for loss in self.tracker.get_regularization_losses().values():
                self.add_loss(loss)
        return out