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
@estimator_export(v1=['estimator.LinearEstimator'])
class LinearEstimator(estimator.Estimator):
    __doc__ = LinearEstimatorV2.__doc__

    def __init__(self, head, feature_columns, model_dir=None, optimizer='Ftrl', config=None, partitioner=None, sparse_combiner='sum', warm_start_from=None):
        """Initializes a `LinearEstimator` instance.

    Args:
      head: A `_Head` instance constructed with a method such as
        `tf.contrib.estimator.multi_label_head`.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` used to train the model. Can also
        be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or
        callable. Defaults to FTRL optimizer.
      config: `RunConfig` object to configure the runtime settings.
      partitioner: Optional. Partitioner for input layer.
      sparse_combiner: A string specifying how to reduce if a categorical column
        is multivalent.  One of "mean", "sqrtn", and "sum" -- these are
        effectively different ways to do example-level normalization, which can
        be useful for bag-of-words features. for more details, see
        `tf.feature_column.linear_model`.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights and biases are warm-started, and it is assumed that vocabularies
        and Tensor names are unchanged.
    """

        def _model_fn(features, labels, mode, config):
            return _linear_model_fn(features=features, labels=labels, mode=mode, head=head, feature_columns=tuple(feature_columns or []), optimizer=optimizer, partitioner=partitioner, config=config, sparse_combiner=sparse_combiner)
        estimator._canned_estimator_api_gauge.get_cell('Estimator').set('Linear')
        super(LinearEstimator, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, warm_start_from=warm_start_from)