from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.DNNLinearCombinedEstimator'])
class DNNLinearCombinedEstimator(estimator.Estimator):
    __doc__ = DNNLinearCombinedEstimatorV2.__doc__

    def __init__(self, head, model_dir=None, linear_feature_columns=None, linear_optimizer='Ftrl', dnn_feature_columns=None, dnn_optimizer='Adagrad', dnn_hidden_units=None, dnn_activation_fn=tf.nn.relu, dnn_dropout=None, input_layer_partitioner=None, config=None, batch_norm=False, linear_sparse_combiner='sum'):
        self._feature_columns = _validate_feature_columns(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns)
        estimator._canned_estimator_api_gauge.get_cell('Estimator').set('DNNLinearCombined')

        def _model_fn(features, labels, mode, config):
            """Call the _dnn_linear_combined_model_fn."""
            return _dnn_linear_combined_model_fn(features=features, labels=labels, mode=mode, head=head, linear_feature_columns=linear_feature_columns, linear_optimizer=linear_optimizer, dnn_feature_columns=dnn_feature_columns, dnn_optimizer=dnn_optimizer, dnn_hidden_units=dnn_hidden_units, dnn_activation_fn=dnn_activation_fn, dnn_dropout=dnn_dropout, input_layer_partitioner=input_layer_partitioner, config=config, batch_norm=batch_norm, linear_sparse_combiner=linear_sparse_combiner)
        super(DNNLinearCombinedEstimator, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)