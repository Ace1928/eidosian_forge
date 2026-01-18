from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import ops
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import FtrlParameters
from tensorflow.python.tpu.tpu_embedding import MomentumParameters
from tensorflow.python.tpu.tpu_embedding import ProximalAdagradParameters
from tensorflow.python.tpu.tpu_embedding import RMSPropParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export(v1=['estimator.tpu.experimental.EmbeddingConfigSpec'])
class EmbeddingConfigSpec(collections.namedtuple('EmbeddingConfigSpec', ['feature_columns', 'tensor_core_feature_columns', 'optimization_parameters', 'clipping_limit', 'pipeline_execution_with_tensor_core', 'experimental_gradient_multiplier_fn', 'feature_to_config_dict', 'table_to_config_dict', 'partition_strategy', 'profile_data_directory'])):
    """Class to keep track of the specification for TPU embeddings.

  Pass this class to `tf.estimator.tpu.TPUEstimator` via the
  `embedding_config_spec` parameter. At minimum you need to specify
  `feature_columns` and `optimization_parameters`. The feature columns passed
  should be created with some combination of
  `tf.tpu.experimental.embedding_column` and
  `tf.tpu.experimental.shared_embedding_columns`.

  TPU embeddings do not support arbitrary Tensorflow optimizers and the
  main optimizer you use for your model will be ignored for the embedding table
  variables. Instead TPU embeddigns support a fixed set of predefined optimizers
  that you can select from and set the parameters of. These include adagrad,
  adam and stochastic gradient descent. Each supported optimizer has a
  `Parameters` class in the `tf.tpu.experimental` namespace.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  column_c = tf.feature_column.categorical_column_with_identity(...)
  tpu_shared_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  tpu_non_shared_column = tf.tpu.experimental.embedding_column(
      column_c, 10)
  tpu_columns = [tpu_non_shared_column] + tpu_shared_columns
  ...
  def model_fn(features):
    dense_features = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_features(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          optimization_parameters=(
              tf.estimator.tpu.experimental.AdagradParameters(0.1))))
  ```

  @compatibility(TF2)
  TPU Estimator manages its own TensorFlow graph and session, so it is not
  compatible with TF2 behaviors. We recommend that you migrate to the newer
  `tf.distribute.TPUStrategy`. See the
  [TPU guide](https://www.tensorflow.org/guide/tpu) for details.
  @end_compatibility
  """

    def __new__(cls, feature_columns=None, optimization_parameters=None, clipping_limit=None, pipeline_execution_with_tensor_core=False, experimental_gradient_multiplier_fn=None, feature_to_config_dict=None, table_to_config_dict=None, partition_strategy='div', profile_data_directory=None):
        """Creates an `EmbeddingConfigSpec` instance.

    Args:
      feature_columns: All embedding `FeatureColumn`s used by model.
      optimization_parameters: An instance of `AdagradParameters`,
        `AdamParameters` or `StochasticGradientDescentParameters`. This
        optimizer will be applied to all embedding variables specified by
        `feature_columns`.
      clipping_limit: (Optional) Clipping limit (absolute value).
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding IDs. Please see
        `tpu_embedding_configuration.proto` for details.
      experimental_gradient_multiplier_fn: (Optional) A Fn taking global step as
        input returning the current multiplier for all embedding gradients.
      feature_to_config_dict: A dictionary mapping feature names to instances of
        the class `FeatureConfig`. Either features_columns or the pair of
        `feature_to_config_dict` and `table_to_config_dict` must be specified.
      table_to_config_dict: A dictionary mapping feature names to instances of
        the class `TableConfig`. Either features_columns or the pair of
        `feature_to_config_dict` and `table_to_config_dict` must be specified.
      partition_strategy: A string, determining how tensors are sharded to the
        tpu hosts. See `tf.nn.safe_embedding_lookup_sparse` for more details.
        Allowed value are `"div"` and `"mod"'. If `"mod"` is used, evaluation
        and exporting the model to CPU will not work as expected.
      profile_data_directory: Directory where embedding lookup statistics are
        stored. These statistics summarize information about the inputs to the
        embedding lookup operation, in particular, the average number of
        embedding IDs per example and how well the embedding IDs are load
        balanced across the system. The lookup statistics are used during TPU
        initialization for embedding table partitioning. Collection of lookup
        statistics is done at runtime by  profiling the embedding inputs, only a
        small fraction of input samples are profiled to minimize host CPU
        overhead. Once a suitable number of samples are profiled, the lookup
        statistics are saved to table-specific files in the profile data
        directory generally at the end of a TPU training loop. The filename
        corresponding to each table is obtained by hashing table specific
        parameters (e.g., table name and number of features) and global
        configuration parameters (e.g., sharding strategy and task count). The
        same profile data directory can be shared among several models to reuse
        embedding lookup statistics.

    Returns:
      An `EmbeddingConfigSpec` instance.

    Raises:
      ValueError: If the feature_columns are not specified.
      TypeError: If the feature columns are not of ths correct type (one of
        _SUPPORTED_FEATURE_COLUMNS, _TPU_EMBEDDING_COLUMN_CLASSES OR
        _EMBEDDING_COLUMN_CLASSES).
      ValueError: If `optimization_parameters` is not one of the required types.
    """
        if not feature_columns and (not (feature_to_config_dict and table_to_config_dict)) or (feature_columns and (feature_to_config_dict and table_to_config_dict)):
            raise ValueError('Exactly one of `feature_columns` and the pair `feature_to_config_dict` and `table_to_config_dict` must be be specified.')
        if partition_strategy not in ('div', 'mod'):
            raise ValueError('Invalid partition_strategy {}. Must be one of "mod" or "div".'.format(partition_strategy))
        tensor_core_feature_columns = None
        embedding_core_feature_columns = None
        if feature_columns:
            tensor_core_feature_columns = []
            embedding_core_feature_columns = []
            supported_classes = tuple(list(_SUPPORTED_FEATURE_COLUMNS) + list(_TPU_EMBEDDING_COLUMN_CLASSES) + list(_EMBEDDING_COLUMN_CLASSES))
            for column in feature_columns:
                if isinstance(column, _TPU_DEVICE_SPECIFIC_EMBEDDING_COLUMNS) and column._embedding_lookup_device == tpu_fc_v2.EmbeddingDevice.TPU_TENSOR_CORE:
                    tensor_core_feature_columns.append(column)
                else:
                    embedding_core_feature_columns.append(column)
                if not isinstance(column, supported_classes):
                    raise TypeError('All feature columns must be supported types in {}. Got {}'.format(supported_classes, type(column)))
            if not isinstance(optimization_parameters, _SUPPORTED_OPTIMIZERS):
                raise ValueError('optimization_parameters must be an instance of type {}. Got {}.'.format(_SUPPORTED_OPTIMIZERS, type(optimization_parameters)))
        else:
            for feature, config in feature_to_config_dict.items():
                if not isinstance(config, tpu_embedding.FeatureConfig):
                    raise TypeError('Config for feature {} must be of type `FeatureConfig`. Got {}'.format(feature, type(config)))
                if config.table_id not in table_to_config_dict:
                    raise ValueError('Feature {} refers to table {} which is not in the table_to_config_dict.'.format(feature, config.table_id))
            for table, config in table_to_config_dict.items():
                if not isinstance(config, tpu_embedding.TableConfig):
                    raise TypeError('Config for table {} must be of type `TableConfig`. Got {}'.format(table, type(config)))
        return super(EmbeddingConfigSpec, cls).__new__(cls, feature_columns=embedding_core_feature_columns, tensor_core_feature_columns=tensor_core_feature_columns, optimization_parameters=optimization_parameters, clipping_limit=clipping_limit, pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core, experimental_gradient_multiplier_fn=experimental_gradient_multiplier_fn, feature_to_config_dict=feature_to_config_dict, table_to_config_dict=table_to_config_dict, partition_strategy=partition_strategy, profile_data_directory=profile_data_directory)