from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
from keras_tuner.src.backend import ops
from keras_tuner.src.backend import random
from keras_tuner.src.backend.keras import layers
from keras_tuner.src.engine import hypermodel
def _build_fixedaug_layers(self, inputs, hp):
    x = inputs
    for transform, (factor_min, factor_max) in self.transforms:
        transform_factor = hp.Float(f'factor_{transform}', factor_min, factor_max, step=0.05, default=factor_min)
        if transform_factor == 0:
            continue
        transform_layer = TRANSFORMS[transform](transform_factor)
        x = transform_layer(x)
    return x