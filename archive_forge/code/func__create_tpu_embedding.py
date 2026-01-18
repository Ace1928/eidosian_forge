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
def _create_tpu_embedding(self, mode):
    """Create tpu_embedding.TPUEmbedding based on mode."""
    if mode == model_fn_lib.ModeKeys.TRAIN:
        batch_size = self._train_batch_size
    else:
        batch_size = self._eval_batch_size
    if mode == model_fn_lib.ModeKeys.TRAIN:
        tpu_embedding_mode = tpu_embedding.TRAINING
        optimization_parameters = self._embedding_config_spec.optimization_parameters
    elif mode == model_fn_lib.ModeKeys.EVAL or mode == model_fn_lib.ModeKeys.PREDICT:
        tpu_embedding_mode = tpu_embedding.INFERENCE
        optimization_parameters = None
    else:
        raise ValueError('Mode {} is not supported.'.format(mode))
    if self._run_config.cluster:
        master = self._run_config.cluster.master()
        cluster_spec = self._run_config.cluster.cluster_spec()
        cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
    else:
        master = self._run_config.evaluation_master if mode == model_fn_lib.ModeKeys.EVAL else self._run_config.master
        cluster_def = None
    master_job_name = None
    if self._run_config.tpu_config.tpu_job_name is not None:
        master_job_name = self._run_config.tpu_config.tpu_job_name
    tpu_embedding_ = tpu_embedding.TPUEmbedding(self._table_to_config_dict, self._feature_to_config_dict, batch_size, tpu_embedding_mode, master, optimization_parameters, cluster_def, pipeline_execution_with_tensor_core=self._embedding_config_spec.pipeline_execution_with_tensor_core, partition_strategy=self._partition_strategy, profile_data_directory=self._embedding_config_spec.profile_data_directory, master_job_name=master_job_name)
    return tpu_embedding_