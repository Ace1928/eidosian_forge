from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import base_delegate
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.util import nest
class LossScaleOptimizer(base_delegate.DelegatingTrackableMixin, optimizer_v2.OptimizerV2):
    """An optimizer that applies loss scaling to prevent numeric underflow.

  Loss scaling is a technique to prevent numeric underflow in intermediate
  gradients when float16 is used. To prevent underflow, the loss is multiplied
  (or "scaled") by a certain factor called the "loss scale", which causes
  intermediate gradients to be scaled by the loss scale as well. The final
  gradients are divided (or "unscaled") by the loss scale to bring them back to
  their original value.

  `LossScaleOptimizer` wraps another optimizer and applies loss scaling to it.
  By default, the loss scale is dynamically updated over time so you do not have
  to choose the loss scale. The `minimize` method automatically scales the loss,
  unscales the gradients, and updates the loss scale so all you have to do is
  wrap your optimizer with a `LossScaleOptimizer` if you use `minimize`. For
  example:

  >>> opt = tf.keras.optimizers.SGD(0.25)
  >>> opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
  >>> var = tf.Variable(1.)
  >>> loss_fn = lambda: var ** 2
  >>> # 'minimize' applies loss scaling and updates the loss sale.
  >>> opt.minimize(loss_fn, var_list=var)
  >>> var.numpy()
  0.5

  If a `tf.GradientTape` is used to compute gradients instead of `minimize`, you
  must scale the loss and gradients manually. This can be done with the
  `LossScaleOptimizer.get_scaled_loss` and
  `LossScaleOptimizer.get_unscaled_gradients` methods. For example:

  >>> with tf.GradientTape() as tape:
  ...   loss = loss_fn()
  ...   scaled_loss = opt.get_scaled_loss(loss)
  >>> scaled_grad = tape.gradient(scaled_loss, var)
  >>> (grad,) = opt.get_unscaled_gradients([scaled_grad])
  >>> opt.apply_gradients([(grad, var)])  # Loss scale is updated here
  >>> var.numpy()
  0.25

  Warning: If you forget to call `get_scaled_loss` or `get_unscaled_gradients`
  (or both) when using a `tf.GradientTape`, the model will likely converge to a
  worse quality. Please make sure you call each function exactly once.

  When mixed precision with float16 is used, there is typically no risk of
  underflow affecting model quality if loss scaling is properly used. See
  [the mixed precision guide](
  https://www.tensorflow.org/guide/keras/mixed_precision) for more information
  on how to use mixed precision.

  Args:
    inner_optimizer: The `tf.keras.optimizers.Optimizer` instance to wrap.
    dynamic: Bool indicating whether dynamic loss scaling is used. Defaults to
      True. If True, the loss scale will be dynamically updated over time using
      an algorithm that keeps the loss scale at approximately its optimal value.
      If False, a single fixed loss scale is used and `initial_scale` must be
      specified, which is used as the loss scale. Recommended to keep as True,
      as choosing a fixed loss scale can be tricky. Currently, there is a small
      performance overhead to dynamic loss scaling compared to fixed loss
      scaling.
    initial_scale: The initial loss scale. If `dynamic` is True, this defaults
      to `2 ** 15`. If `dynamic` is False, this must be specified and acts as
      the sole loss scale, as the loss scale does not change over time. When
      dynamic loss scaling is used, is better for this to be a very high number,
      because a loss scale that is too high gets lowered far more quickly than a
      loss scale that is too low gets raised.
    dynamic_growth_steps: With dynamic loss scaling, every
      `dynamic_growth_steps` steps with finite gradients, the loss scale is
      doubled. Defaults to 2000. If a nonfinite gradient is encountered, the
      count is reset back to zero, gradients are skipped that step, and the loss
      scale is halved. The count can be queried with
      `LossScaleOptimizer.dynamic_counter`. This argument can only be specified
      if `dynamic` is True.

  `LossScaleOptimizer` will occasionally skip applying gradients to the
  variables, in which case the trainable variables will not change that step.
  This is done because the dynamic loss scale will sometimes be raised too
  high, causing overflow in the gradients. Typically, the first 2 to 15 steps of
  the model are skipped as the initial loss scale is very high, but afterwards
  steps will only be skipped on average 0.05% of the time (the fraction of steps
  skipped is `1 / dynamic_growth_steps`).

  `LossScaleOptimizer` delegates all public `Optimizer` methods to the inner
  optimizer. Additionally, in methods `minimize` and `get_gradients`, it scales
  the loss and unscales the gradients. In methods `minimize` and
  `apply_gradients`, it additionally updates the loss scale and skips applying
  gradients if any gradient has a nonfinite value.

  ### Hyperparameters

  Hyperparameters can be accessed and set on the LossScaleOptimizer, which will
  be delegated to the wrapped optimizer.

  >>> opt = tf.keras.optimizers.Adam(beta_1=0.8, epsilon=1e-5)
  >>> opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
  >>> opt.beta_1  # Equivalent to `opt.inner_optimizer.beta_1`
  0.8
  >>> opt.beta_1 = 0.7  # Equivalent to `opt.inner_optimizer.beta_1 = 0.7`
  >>> opt.beta_1
  0.7
  >>> opt.inner_optimizer.beta_1
  0.7

  However, accessing or setting non-hyperparameters is not delegated to the
  LossScaleOptimizer. In an Adam optimizer, `beta_1` is a hyperparameter but
  `epsilon` is not, as the Adam optimizer only calls `Optimizer._set_hyper` on
  `beta_1`.

  >>> opt.inner_optimizer.epsilon
  1e-5
  >>> opt.epsilon
  Traceback (most recent call last):
  ...
  AttributeError: 'LossScaleOptimizer' object has no attribute 'epsilon'
  >>> opt.epsilon = 1e-4  # This does NOT set epsilon on `opt.inner_optimizer`
  >>> opt.inner_optimizer.epsilon
  >>> 1e-5

  In the above example, despite epsilon being set on the LossScaleOptimizer, the
  old epsilon value will still be used when training as epsilon was not set on
  the inner optimizer.
  """
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None):
        if not isinstance(inner_optimizer, optimizer_v2.OptimizerV2):
            raise TypeError('"inner_optimizer" must be an instance of OptimizerV2, but got: %s' % inner_optimizer)
        if not isinstance(dynamic, bool):
            raise TypeError('"dynamic" argument to LossScaleOptimizer.__init__ must be a bool, but got: %r' % (dynamic,))
        if isinstance(inner_optimizer, LossScaleOptimizer):
            raise TypeError('LossScaleOptimizer cannot wrap another LossScaleOptimizer, but got: %s' % (inner_optimizer,))
        self._raise_if_strategy_unsupported()
        if getattr(inner_optimizer, '_is_wrapped_by_loss_scale_optimizer', False):
            raise ValueError('"inner_optimizer" is already wrapped by a LossScaleOptimizer. An optimizer can only be wrapped by a single LossScaleOptimizer')
        self._optimizer = inner_optimizer
        self._optimizer._is_wrapped_by_loss_scale_optimizer = True
        base_delegate.DelegatingTrackableMixin.__init__(self, self._optimizer)
        if dynamic:
            if initial_scale is None:
                initial_scale = _DEFAULT_INITIAL_SCALE
            if dynamic_growth_steps is None:
                dynamic_growth_steps = _DEFAULT_GROWTH_STEPS
            self._loss_scale = _DynamicLossScaleState(initial_scale, dynamic_growth_steps, multiplier=2)
            self._track_trackable(self._loss_scale, 'loss_scale')
        else:
            if initial_scale is None:
                raise ValueError('"initial_scale" must be specified if "dynamic" is False')
            self._loss_scale = float(initial_scale)
            if dynamic_growth_steps is not None:
                raise ValueError('"dynamic_growth_steps" must be None if "dynamic" is False, but got: %s' % (dynamic_growth_steps,))
        self._track_trackable(FakeOptimizerForRestoration(self._optimizer), 'base_optimizer')

    @property
    def dynamic(self):
        """Bool indicating whether dynamic loss scaling is used."""
        return isinstance(self._loss_scale, _DynamicLossScaleState)

    @property
    def loss_scale(self):
        """The current loss scale as a float32 scalar tensor."""
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(self._loss_scale.current_loss_scale)
        else:
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(self._loss_scale)

    @property
    def dynamic_counter(self):
        """The number of steps since the loss scale was last increased or decreased.

    This is None if `LossScaleOptimizer.dynamic` is False.

    The counter is incremented every step. Once it reaches
    `LossScaleOptimizer.dynamic_growth_steps`, the loss scale will be doubled
    and the counter will be reset back to zero. If nonfinite gradients are
    encountered, the loss scale will be halved and the counter will be reset
    back to zero.
    """
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return self._loss_scale.counter
        else:
            return None

    @property
    def initial_scale(self):
        """The initial loss scale.

    If `LossScaleOptimizer.dynamic` is False, this is the same number as
    `LossScaleOptimizer.loss_scale`, as the loss scale never changes.
    """
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return self._loss_scale.initial_loss_scale
        else:
            return self._loss_scale

    @property
    def dynamic_growth_steps(self):
        """The number of steps it takes to increase the loss scale.

    This is None if `LossScaleOptimizer.dynamic` is False.

    Every `dynamic_growth_steps` consecutive steps with finite gradients, the
    loss scale is increased.
    """
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return self._loss_scale.growth_steps
        else:
            return None

    @property
    def inner_optimizer(self):
        """The optimizer that this LossScaleOptimizer is wrapping."""
        return self._optimizer

    def get_scaled_loss(self, loss):
        """Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale`.
    """
        if callable(loss):

            def new_loss():
                loss_val = loss()
                return loss_val * math_ops.cast(self.loss_scale, loss_val.dtype)
            return new_loss
        else:
            return loss * math_ops.cast(self.loss_scale, loss.dtype)

    def get_unscaled_gradients(self, grads):
        """Unscales the gradients by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.LossScaleOptimizer` doc for an
    example.

    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.

    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale`.
    """
        loss_scale_reciprocal = 1.0 / self.loss_scale
        return [_multiply_gradient(g, loss_scale_reciprocal) if g is not None else None for g in grads]

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
        tape = backprop.GradientTape() if tape is None else tape
        with tape:
            loss = self.get_scaled_loss(loss)
        grads_and_vars = self._optimizer._compute_gradients(loss, var_list, grad_loss, tape=tape)
        grads = [g for g, _ in grads_and_vars]
        weights = [v for _, v in grads_and_vars]
        unscaled_grads = self.get_unscaled_gradients(grads)
        return list(zip(unscaled_grads, weights))

    def get_gradients(self, loss, params):
        loss = self.get_scaled_loss(loss)
        grads = self._optimizer.get_gradients(loss, params)
        return self.get_unscaled_gradients(grads)

    def _create_all_weights(self, var_list):
        self._optimizer._create_all_weights(var_list)

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        if distribute_lib.in_cross_replica_context():
            raise ValueError('apply_gradients() must be called in a replica context.')
        self._raise_if_strategy_unsupported()
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        if experimental_aggregate_gradients:
            grads_and_vars = self._optimizer._transform_unaggregated_gradients(grads_and_vars)
            grads_and_vars = self._optimizer._aggregate_gradients(grads_and_vars)
        grads_and_vars = tuple(grads_and_vars)
        grads = [g for g, _ in grads_and_vars]
        wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])

        def do_not_apply_fn():
            return self._optimizer.iterations.assign_add(1, read_value=False)

        def _if_should_apply_grads(grads):
            if isinstance(self._loss_scale, _DynamicLossScaleState):
                return self._loss_scale.update(grads)
            else:
                return (control_flow_ops.no_op(), True)
        if optimizer_utils.strategy_supports_no_merge_call():
            loss_scale_update_op, should_apply_grads = _if_should_apply_grads(grads)

            def apply_fn():
                return self._apply_gradients(grads, wrapped_vars, name)
            maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn, do_not_apply_fn)
            return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)
        else:

            def _apply_gradients_cross_replica(distribution, grads, wrapped_vars, name):
                loss_scale_update_op, should_apply_grads = _if_should_apply_grads(grads)

                def apply_fn():
                    return distribution.extended.call_for_each_replica(self._apply_gradients, args=(grads, wrapped_vars, name))
                maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn, do_not_apply_fn)
                return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)
            return distribute_lib.get_replica_context().merge_call(_apply_gradients_cross_replica, args=(grads, wrapped_vars, name))

    def _apply_gradients(self, grads, wrapped_vars, name):
        return self._optimizer.apply_gradients(list(zip(grads, wrapped_vars.value)), name, experimental_aggregate_gradients=False)

    def get_config(self):
        serialized_optimizer = optimizers.serialize(self._optimizer)
        return {'inner_optimizer': serialized_optimizer, 'dynamic': self.dynamic, 'initial_scale': self.initial_scale, 'dynamic_growth_steps': self.dynamic_growth_steps}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        if 'loss_scale' in config:
            loss_scale = keras_loss_scale_module.deserialize(config.pop('loss_scale'))
            if isinstance(loss_scale, loss_scale_module.FixedLossScale):
                config['dynamic'] = False
                config['initial_scale'] = loss_scale._loss_scale_value
            elif isinstance(loss_scale, loss_scale_module.DynamicLossScale):
                config['dynamic'] = True
                config['initial_scale'] = loss_scale.initial_loss_scale
                config['dynamic_growth_steps'] = loss_scale.increment_period
                if loss_scale.multiplier != 2:
                    raise ValueError('Cannot deserialize LossScaleOptimizer with a DynamicLossScale whose multiplier is not 2. Got DynamicLossScale: %s' % (loss_scale,))
            else:
                raise ValueError('Serialized LossScaleOptimizers with a LossScale that is neither a FixedLossScale nor a DynamicLossScale can no longer be deserialized')
            config['inner_optimizer'] = config.pop('optimizer')
        config['inner_optimizer'] = optimizers.deserialize(config['inner_optimizer'], custom_objects=custom_objects)
        return cls(**config)

    def _raise_if_strategy_unsupported(self):
        if not strategy_supports_loss_scaling():
            strategy = distribute_lib.get_strategy()
            if isinstance(strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1, tpu_strategy.TPUStrategyV2)):
                raise ValueError('Loss scaling is not supported with TPUStrategy. Loss scaling is unnecessary with TPUs, since they support bfloat16 instead of float16 and bfloat16 does not require loss scaling. You should remove the use of the LossScaleOptimizer when TPUs are used.')
            else:
                raise ValueError('Loss scaling is not supported with the tf.distribute.Strategy: %s. Try using a different Strategy, e.g. a MirroredStrategy' % strategy.__class__.__name__)

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

    def get_slot_names(self):
        return self._optimizer.get_slot_names()

    def variables(self):
        return self._optimizer.variables()

    @property
    def weights(self):
        return self._optimizer.weights

    def get_weights(self):
        return self._optimizer.get_weights()

    def set_weights(self, weights):
        return self._optimizer.set_weights(weights)

    @property
    def clipnorm(self):
        return self._optimizer.clipnorm

    @clipnorm.setter
    def clipnorm(self, val):
        self._optimizer.clipnorm = val

    @property
    def global_clipnorm(self):
        return self._optimizer.global_clipnorm

    @global_clipnorm.setter
    def global_clipnorm(self, val):
        self._optimizer.global_clipnorm = val

    @property
    def clipvalue(self):
        return self._optimizer.clipvalue

    @clipvalue.setter
    def clipvalue(self, val):
        self._optimizer.clipvalue = val

    def _aggregate_gradients(self, grads_and_vars):
        return self._optimizer._aggregate_gradients(grads_and_vars)

    def _restore_slot_variable(self, slot_name, variable, slot_variable):
        return self._optimizer._restore_slot_variable(slot_name, variable, slot_variable)

    def _create_or_restore_slot_variable(self, slot_variable_position, slot_name, variable):
        return self._optimizer._create_or_restore_slot_variable(slot_variable_position, slot_name, variable)

    def get_slot(self, var, slot_name):
        return self._optimizer.get_slot(var, slot_name)

    def add_slot(self, var, slot_name, initializer='zeros'):
        return self._optimizer.add_slot(var, slot_name, initializer)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name == '_optimizer' or name == '_hyper':
                raise e
            if name == 'lr':
                name = 'learning_rate'
            if name in self._optimizer._hyper:
                return self._optimizer._get_hyper(name)
            raise e

    def __dir__(self):
        result = set(super(LossScaleOptimizer, self).__dir__())
        if '_optimizer' in result:
            result |= self._optimizer._hyper.keys()
            if 'learning_rate' in self._optimizer._hyper.keys():
                result.add('lr')
        return list(result)

    def __setattr__(self, name, value):
        if name == 'lr':
            name = 'learning_rate'
        try:
            if name != 'iterations':
                object.__getattribute__(self, name)
            has_attribute = True
        except AttributeError:
            has_attribute = False
        if name != '_optimizer' and name in self._optimizer._hyper and (not has_attribute):
            self._optimizer._set_hyper(name, value)
        else:
            super(LossScaleOptimizer, self).__setattr__(name, value)

    @property
    def learning_rate(self):
        return self._optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._optimizer.learning_rate = value

    @property
    def lr(self):
        return self._optimizer.learning_rate

    @lr.setter
    def lr(self, value):
        self._optimizer.lr = value