import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class PandasSliceable(Sliceable):

    def __getitem__(self, indices):
        return self.array.iloc[indices]

    @classmethod
    def convert_to_numpy(cls, x):
        return x.to_numpy()

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        return cls.convert_to_numpy(x)

    @classmethod
    def convert_to_jax_compatible(cls, x):
        return cls.convert_to_numpy(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        return cls.convert_to_numpy(x)