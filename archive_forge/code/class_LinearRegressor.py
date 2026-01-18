from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.LinearRegressor'])
class LinearRegressor(estimator.Estimator):
    __doc__ = LinearRegressorV2.__doc__.replace('SUM_OVER_BATCH_SIZE', 'SUM')

    def __init__(self, feature_columns, model_dir=None, label_dimension=1, weight_column=None, optimizer='Ftrl', config=None, partitioner=None, warm_start_from=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM, sparse_combiner='sum'):
        _validate_linear_sdca_optimizer_for_linear_regressor(feature_columns=feature_columns, label_dimension=label_dimension, optimizer=optimizer, sparse_combiner=sparse_combiner)
        head = head_lib._regression_head(label_dimension=label_dimension, weight_column=weight_column, loss_reduction=loss_reduction)
        estimator._canned_estimator_api_gauge.get_cell('Regressor').set('Linear')

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _linear_model_fn."""
            return _linear_model_fn(features=features, labels=labels, mode=mode, head=head, feature_columns=tuple(feature_columns or []), optimizer=optimizer, partitioner=partitioner, config=config, sparse_combiner=sparse_combiner)
        super(LinearRegressor, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, warm_start_from=warm_start_from)