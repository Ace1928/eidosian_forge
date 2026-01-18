import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
Save the `FeatureSpace` instance to a `.keras` file.

        You can reload it via `keras.models.load_model()`:

        ```python
        feature_space.save("featurespace.keras")
        reloaded_fs = keras.models.load_model("featurespace.keras")
        ```
        