from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
from keras_tuner.src.backend import ops
from keras_tuner.src.backend import random
from keras_tuner.src.backend.keras import layers
from keras_tuner.src.engine import hypermodel
def _build_randaug_layers(self, inputs, hp):
    augment_layers = hp.Int('augment_layers', self.augment_layers_min, self.augment_layers_max, default=self.augment_layers_min)
    x = inputs
    for _ in range(augment_layers):
        factors = []
        for i, (transform, (f_min, f_max)) in enumerate(self.transforms):
            factor = hp.Float(f'factor_{transform}', f_min, f_max, default=f_min)
            factors.append(factor)
        x = RandAugment([layer for layer, _ in self.transforms], factors)(x)
    return x