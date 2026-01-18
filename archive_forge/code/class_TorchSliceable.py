import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class TorchSliceable(Sliceable):

    @classmethod
    def cast(cls, x, dtype):
        from keras.src.backend.torch.core import cast
        return cast(x, dtype)

    @classmethod
    def convert_to_numpy(cls, x):
        from keras.src.backend.torch.core import convert_to_numpy
        return convert_to_numpy(x)