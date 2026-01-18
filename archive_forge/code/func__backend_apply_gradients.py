import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
def _backend_apply_gradients(self, grads, trainable_variables):
    """Apply method that can be overridden by different backends.

        JAX overrides it in order to deal with statelessness in gradient
        accumulation and EMA handling.

        The below implementation is intended to be generally backend-agnostic,
        but may not work with all backends.

        This method does 4 things:
        - Call the optimizer's update_step() to update trainable variables
            and optimizer variables.
        - Update EMA variables, if EMA is configured.
        - Update gradient accumulators, if gradient accumulation is configured.
        - Update the iteration counter.
        """
    if self.gradient_accumulation_steps:
        is_update_step = (self.iterations + 1) % self.gradient_accumulation_steps == 0

        def _update_step_fn(self, grads, trainable_variables):
            steps = self.gradient_accumulation_steps
            grads = [(grads[i] + self._accumulated_gradients[i]) / steps for i in range(len(grads))]
            self._backend_update_step(grads, trainable_variables, self.learning_rate)
            self._backend_reset_gradient_accumulators()

        def _grad_accumulation_fn(self, grads):
            self._backend_increment_gradient_accumulators(grads)
        ops.cond(is_update_step, lambda: _update_step_fn(self, grads, trainable_variables), lambda: _grad_accumulation_fn(self, grads))
    else:
        self._backend_update_step(grads, trainable_variables, self.learning_rate)
    if self.use_ema:
        self._update_model_variables_moving_average(self._trainable_variables)
        if self.ema_overwrite_frequency:
            should_overwrite_model_vars = (self.iterations + 1) % self.ema_overwrite_frequency == 0
            ops.cond(should_overwrite_model_vars, lambda: self._overwrite_model_variables_with_average_value(self._trainable_variables), lambda: None)
    self.iterations.assign_add(1)