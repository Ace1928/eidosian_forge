import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module
class ImageAdapter(adapter_module.Adapter):

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to ImageInput to be numpy.ndarray or tf.data.Dataset, but got {type}.'.format(type=type(x)))
        if isinstance(x, np.ndarray) and (not np.issubdtype(x.dtype, np.number)):
            raise TypeError('Expect the data to ImageInput to be numerical, but got {type}.'.format(type=x.dtype))