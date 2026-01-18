from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
class TFOptimizer(Optimizer, trackable.Trackable):
    """Wrapper class for native TensorFlow optimizers."""

    def __init__(self, optimizer, iterations=None):
        self.optimizer = optimizer
        self._track_trackable(optimizer, name='optimizer')
        if iterations is None:
            with backend.name_scope(self.__class__.__name__):
                self.iterations = backend.variable(0, dtype='int64', name='iterations')
        else:
            self.iterations = iterations
        self._track_trackable(self.iterations, name='global_step')

    def _clip_gradients(self, grads):
        """Clip gradients according to the clipnorm and clipvalue attributes."""
        return grads

    def minimize(self, loss, var_list, grad_loss=None, tape=None):
        """Mimics the `OptimizerV2.minimize` API."""
        if not callable(loss) and tape is None:
            raise ValueError('`tape` is required when a `Tensor` loss is passed.')
        tape = tape if tape is not None else backprop.GradientTape()
        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()
        var_list = nest.flatten(var_list)
        if var_list:
            grads = tape.gradient(loss, var_list, grad_loss)
            grads_and_vars = list(zip(grads, var_list))
            self.apply_gradients(grads_and_vars)

    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars, global_step=self.iterations)

    def get_grads(self, loss, params):
        return self.optimizer.compute_gradients(loss, params)

    def get_updates(self, loss, params):
        if distribute_lib.has_strategy():
            self.updates = []
            if not params:
                grads = self.optimizer.compute_gradients(loss)
            else:
                grads = self.optimizer.compute_gradients(loss, params)
            global_step = training_util.get_global_step()
            opt_update = self.optimizer.apply_gradients(grads, global_step)
        else:
            if not params:
                self.updates = [state_ops.assign_add(self.iterations, 1)]
                return self.updates
            self.updates = []
            grads = self.optimizer.compute_gradients(loss, params)
            opt_update = self.optimizer.apply_gradients(grads, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    @property
    def weights(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def from_config(self, config):
        raise NotImplementedError