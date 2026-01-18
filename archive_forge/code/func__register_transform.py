from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
from keras_tuner.src.backend import ops
from keras_tuner.src.backend import random
from keras_tuner.src.backend.keras import layers
from keras_tuner.src.engine import hypermodel
def _register_transform(self, transform_name, transform_params):
    """Register a transform and format parameters for tuning the transform.

        Args:
            transform_name: A string, the name of the transform.
            trnasform_params: A number between [0, 1], a list of two numbers
                between [0, 1] or None. If set to a single number x, the
                corresponding transform factor will be between [0, x].
                If set to a list of 2 numbers [x, y], the factor will be
                between [x, y]. If set to None, the transform will be excluded.
        """
    if not transform_params:
        return
    try:
        transform_factor_min = transform_params[0]
        transform_factor_max = transform_params[1]
        if len(transform_params) > 2:
            raise ValueError(f'Length of keyword argument {transform_name} must not exceed 2.')
    except TypeError:
        transform_factor_min = 0
        transform_factor_max = transform_params
    if not (isinstance(transform_factor_max, (int, float)) and isinstance(transform_factor_min, (int, float))):
        raise ValueError(f'Keyword argument {transform_name} must be int or float, but received {transform_params}.')
    self.transforms.append((transform_name, (transform_factor_min, transform_factor_max)))