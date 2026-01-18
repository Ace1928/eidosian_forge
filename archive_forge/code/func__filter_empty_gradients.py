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
def _filter_empty_gradients(self, grads, vars):
    for grad in grads:
        if grad is None:
            filtered = [(g, v) for g, v in zip(grads, vars) if g is not None]
            if not filtered:
                raise ValueError('No gradients provided for any variable.')
            if len(filtered) < len(grads):
                missing_grad_vars = [v for g, v in zip(grads, vars) if g is None]
                warnings.warn(f'Gradients do not exist for variables {[v.name for v in missing_grad_vars]} when minimizing the loss. If using `model.compile()`, did you forget to provide a `loss` argument?')
            return zip(*filtered)
    return (grads, vars)