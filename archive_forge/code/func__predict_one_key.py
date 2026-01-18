from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def _predict_one_key(self, input_fn, predict_key):
    for result in self.predict(input_fn=input_fn, predict_keys=[predict_key]):
        yield result[predict_key]