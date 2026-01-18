from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _stateless_handle_finite_grads(self, optimizer_variables, grads, trainable_variables):

    def upscale():
        mapping = list(zip(self.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.step_counter.assign(0)
            self.dynamic_scale.assign(self.dynamic_scale * 2.0)
        return [scope.get_current_value(v) for v in self._variables]

    def increment():
        mapping = list(zip(self.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.step_counter.assign_add(1)
        return [scope.get_current_value(v) for v in self._variables]
    mapping = list(zip(self.variables, optimizer_variables))
    with backend.StatelessScope(state_mapping=mapping):
        own_variables = ops.cond(ops.equal(self.step_counter, self.dynamic_growth_steps - 1), upscale, increment)
        scale = self.dynamic_scale
        unscaled_grads = [g if g is None else ops.divide(g, scale) for g in grads]
        new_trainable_variables, new_inner_variables = self.inner_optimizer.stateless_apply(self.inner_optimizer.variables, unscaled_grads, trainable_variables)
    new_optimizer_variables = own_variables + new_inner_variables
    return (new_trainable_variables, new_optimizer_variables)