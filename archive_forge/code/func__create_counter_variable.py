import copy
import itertools
import json
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers
from keras.src.dtensor import dtensor_api
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import compile_utils
from keras.src.engine import data_adapter
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import steps_per_execution_tuning
from keras.src.engine import training_utils
from keras.src.metrics import base_metric
from keras.src.mixed_precision import loss_scale_optimizer as lso
from keras.src.optimizers import optimizer
from keras.src.optimizers import optimizer_v1
from keras.src.saving import pickle_utils
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import model_serialization
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.mode_keys import ModeKeys
def _create_counter_variable(self, init_value):
    """Helper function for counter variable creation.

        For the DTensor use case with layout map, since the variable are not
        tracked by model, they can't be visited by the layout map, and need to
        be properly initialized as DVariable.
        """
    if self._layout_map is None:
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        return tf.Variable(init_value, dtype='int64', aggregation=agg)
    else:
        layout = dtensor_api.Layout.replicated(mesh=self._layout_map.get_default_mesh(), rank=0)
        return dtensor_api.DVariable(init_value, dtype='int64', layout=layout)