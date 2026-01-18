import math
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils.dataset_utils import is_torch_tensor
from keras.src.utils.nest import lists_to_tuples
def __getitems__(self, indices):

    def slice_and_convert(x):
        return convert_to_tensor(np.take(x, indices, axis=0))
    return tree.map_structure(slice_and_convert, self.array)