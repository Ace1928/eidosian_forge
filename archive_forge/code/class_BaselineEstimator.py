from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_v1
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.BaselineEstimator'])
class BaselineEstimator(estimator.Estimator):
    __doc__ = BaselineEstimatorV2.__doc__

    def __init__(self, head, model_dir=None, optimizer='Ftrl', config=None):

        def _model_fn(features, labels, mode, config):
            return _baseline_model_fn(features=features, labels=labels, mode=mode, head=head, optimizer=optimizer, config=config)
        super(BaselineEstimator, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)