from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.DNNEstimator'])
class DNNEstimator(estimator.Estimator):
    __doc__ = DNNEstimatorV2.__doc__

    def __init__(self, head, hidden_units, feature_columns, model_dir=None, optimizer='Adagrad', activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=None, config=None, warm_start_from=None, batch_norm=False):

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _dnn_model_fn."""
            return _dnn_model_fn(features=features, labels=labels, mode=mode, head=head, hidden_units=hidden_units, feature_columns=tuple(feature_columns or []), optimizer=optimizer, activation_fn=activation_fn, dropout=dropout, input_layer_partitioner=input_layer_partitioner, config=config, batch_norm=batch_norm)
        estimator._canned_estimator_api_gauge.get_cell('Estimator').set('DNN')
        super(DNNEstimator, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, warm_start_from=warm_start_from)