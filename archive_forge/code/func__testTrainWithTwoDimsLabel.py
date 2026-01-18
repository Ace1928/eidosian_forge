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
def _testTrainWithTwoDimsLabel(self, n_classes):
    batch_size = 20
    est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    data_rank_2 = np.array([[0], [1]])
    self.assertEqual((2,), data_rank_1.shape)
    self.assertEqual((2, 1), data_rank_2.shape)
    train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1}, y=data_rank_2, batch_size=batch_size, num_epochs=None, shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(n_classes, 200)