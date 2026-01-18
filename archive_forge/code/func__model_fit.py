import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src import callbacks as callbacks_lib
from keras.src.engine import sequential
from keras.src.layers import core as core_layers
from keras.src.layers.preprocessing import string_lookup
from keras.src.optimizers.legacy import gradient_descent
from keras.src.utils import dataset_creator
from tensorflow.python.platform import tf_logging as logging
def _model_fit(self, strategy, steps_per_execution=1, validation_data=None, x=None, y=None, shuffle=True, batch_size=None, steps_per_epoch=10, run_eagerly=False, with_normalization_layer=False, callbacks=None, use_lookup_layer=False, use_dataset_creator=True, verbose='auto', jit_compile=None):
    if callbacks is None:
        callbacks = []
    model, default_callbacks = self._model_compile(strategy, steps_per_execution, run_eagerly, with_normalization_layer, jit_compile)
    callbacks += default_callbacks
    if x is None:
        if use_dataset_creator:
            x = dataset_creator.DatasetCreator(self._get_dataset_fn(use_lookup_layer))
        else:
            x = self._get_dataset_fn(use_lookup_layer)(None)
    if validation_data is None:
        if use_dataset_creator:
            validation_data = dataset_creator.DatasetCreator(self._get_dataset_fn(use_lookup_layer))
        else:
            validation_data = self._get_dataset_fn(use_lookup_layer)(None)
    model.fit(x, y, shuffle=shuffle, batch_size=batch_size, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=validation_data, validation_steps=steps_per_epoch, verbose=verbose)
    return model