import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class TensorflowSliceable(Sliceable):

    def __getitem__(self, indices):
        from keras.src.utils.module_utils import tensorflow as tf
        if isinstance(indices, slice):
            return self.array[indices]
        else:
            return tf.gather(self.array, indices, axis=0)

    @classmethod
    def cast(cls, x, dtype):
        from keras.src.backend.tensorflow.core import cast
        return cast(x, dtype)

    @classmethod
    def convert_to_numpy(cls, x):
        from keras.src.backend.tensorflow.core import convert_to_numpy
        return convert_to_numpy(x)