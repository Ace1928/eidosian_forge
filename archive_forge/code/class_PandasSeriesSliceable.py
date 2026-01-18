import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class PandasSeriesSliceable(PandasSliceable):

    @classmethod
    def convert_to_numpy(cls, x):
        return np.expand_dims(x.to_numpy(), axis=-1)