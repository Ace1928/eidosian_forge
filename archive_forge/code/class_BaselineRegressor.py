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
@estimator_export(v1=['estimator.BaselineRegressor'])
class BaselineRegressor(estimator.Estimator):
    __doc__ = BaselineRegressorV2.__doc__.replace('SUM_OVER_BATCH_SIZE', 'SUM')

    def __init__(self, model_dir=None, label_dimension=1, weight_column=None, optimizer='Ftrl', config=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM):
        head = head_lib._regression_head(label_dimension=label_dimension, weight_column=weight_column, loss_reduction=loss_reduction)

        def _model_fn(features, labels, mode, config):
            return _baseline_model_fn(features=features, labels=labels, mode=mode, head=head, optimizer=optimizer, config=config)
        super(BaselineRegressor, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)