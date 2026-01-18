import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine import base_preprocessing_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import tf_utils
class PreprocessingStage(sequential.Sequential, base_preprocessing_layer.PreprocessingLayer):
    """A sequential preprocessing stage.

    This preprocessing stage wraps a list of preprocessing layers into a
    Sequential-like object that enables you to `adapt()` the whole list via
    a single `adapt()` call on the preprocessing stage.

    Args:
      layers: List of layers. Can include layers that aren't preprocessing
        layers.
      name: String. Optional name for the preprocessing stage object.
    """

    def adapt(self, data, reset_state=True):
        """Adapt the state of the layers of the preprocessing stage to the data.

        Args:
          data: A batched Dataset object, or a NumPy array, or an EagerTensor.
            Data to be iterated over to adapt the state of the layers in this
            preprocessing stage.
          reset_state: Whether this call to `adapt` should reset the state of
            the layers in this preprocessing stage.
        """
        if not isinstance(data, (tf.data.Dataset, np.ndarray, tf.__internal__.EagerTensor)):
            raise ValueError(f'`adapt()` requires a batched Dataset, an EagerTensor, or a Numpy array as input. Received data={data}')
        if isinstance(data, tf.data.Dataset):
            if tf_utils.dataset_is_infinite(data):
                raise ValueError('The dataset passed to `adapt()` has an infinite number of elements. Please use dataset.take(...) to make the number of elements finite.')
        for current_layer_index in range(0, len(self.layers)):
            if not hasattr(self.layers[current_layer_index], 'adapt'):
                continue

            def map_fn(x):
                """Maps this object's inputs to those at current_layer_index.

                Args:
                  x: Batch of inputs seen in entry of the `PreprocessingStage`
                    instance.

                Returns:
                  Batch of inputs to be processed by layer
                    `self.layers[current_layer_index]`
                """
                if current_layer_index == 0:
                    return x
                for i in range(current_layer_index):
                    x = self.layers[i](x)
                return x
            if isinstance(data, tf.data.Dataset):
                current_layer_data = data.map(map_fn)
            else:
                current_layer_data = map_fn(data)
            self.layers[current_layer_index].adapt(current_layer_data, reset_state=reset_state)