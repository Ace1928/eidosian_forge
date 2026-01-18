import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _build_index_dict(self, var_list):
    """Build variable to index dictionary.

        Build a dictionary that maps variable to the index of it in the given
        var_list.

        Args:
          var_list: List of variables to build index dict on.

        Returns:
          None
        """
    self._index_dict = {}
    for i, var in enumerate(var_list):
        var_key = self._var_key(var)
        self._index_dict[var_key] = i