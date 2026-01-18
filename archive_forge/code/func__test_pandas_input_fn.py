from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def _test_pandas_input_fn(self, n_classes):
    """Tests complete flow with pandas_input_fn."""
    if not HAS_PANDAS:
        return
    input_dimension = 1
    batch_size = 10
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    target = np.array([1, 0, 1, 0], dtype=np.int32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(target)
    prediction_length = 4
    train_input_fn = pandas_io.pandas_input_fn(x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(x=x, y=y, batch_size=batch_size, shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(x=x, batch_size=batch_size, shuffle=False)
    self._test_complete_flow(n_classes=n_classes, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, predict_input_fn=predict_input_fn, input_dimension=input_dimension, prediction_length=prediction_length)