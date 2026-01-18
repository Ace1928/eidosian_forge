import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class ScipySparseSliceable(Sliceable):

    def __init__(self, array):
        super().__init__(array.tocsr())

    @classmethod
    def convert_to_numpy(cls, x):
        return x.todense()

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        return to_tensorflow_sparse_wrapper(data_adapter_utils.scipy_sparse_to_tf_sparse(x))

    @classmethod
    def convert_to_jax_compatible(cls, x):
        return data_adapter_utils.scipy_sparse_to_jax_sparse(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        return x.todense()