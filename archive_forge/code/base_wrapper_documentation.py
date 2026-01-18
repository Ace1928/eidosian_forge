import copy
from keras.src.engine.base_layer import Layer
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization as legacy_serialization
from tensorflow.python.util.tf_export import keras_export
Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    Args:
      layer: The layer to be wrapped.
    