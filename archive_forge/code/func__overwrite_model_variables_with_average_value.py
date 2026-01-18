import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def _overwrite_model_variables_with_average_value(self, trainable_variables):
    """Overwrite model variables with their moving average values.

        This function overwrites variables on each device.
        Args:
          var_list: list of model variables.
        """
    trainable_variables = [v.value if isinstance(v, backend.Variable) else v for v in trainable_variables]
    for var, average_var in zip(trainable_variables, self._model_variables_moving_average):
        self._distribution_strategy.extended.update(var, lambda a, b: a.assign(b), args=(average_var,))