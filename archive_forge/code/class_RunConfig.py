from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.tpu import util as util_lib
@estimator_export(v1=['estimator.tpu.RunConfig'])
class RunConfig(run_config_lib.RunConfig):
    """RunConfig with TPU support."""

    def __init__(self, tpu_config=None, evaluation_master=None, master=None, cluster=None, **kwargs):
        """Constructs a RunConfig.

    Args:
      tpu_config: the TPUConfig that specifies TPU-specific configuration.
      evaluation_master: a string. The address of the master to use for eval.
        Defaults to master if not set.
      master: a string. The address of the master to use for training.
      cluster: a ClusterResolver
      **kwargs: keyword config parameters.

    Raises:
      ValueError: if cluster is not None and the provided session_config has a
        cluster_def already.

    @compatibility(TF2)
    TPU Estimator manages its own TensorFlow graph and session, so it is not
    compatible with TF2 behaviors. We recommend that you migrate to the newer
    `tf.distribute.TPUStrategy`. See the
    [TPU guide](https://www.tensorflow.org/guide/tpu) for details.
    @end_compatibility
    """
        super(RunConfig, self).__init__(**kwargs)
        self._tpu_config = tpu_config or TPUConfig()
        self._cluster = cluster
        if master is not None:
            if cluster is not None:
                raise ValueError('Both master and cluster are set.')
            self._master = master
        elif cluster:
            self._master = cluster.master()
        if evaluation_master is not None:
            self._evaluation_master = evaluation_master
        elif not self._evaluation_master and self.task_type != run_config_lib.TaskType.EVALUATOR:
            self._evaluation_master = self._master
        if cluster:
            self._cluster_spec = cluster.cluster_spec()
            if self._session_config is None:
                self._session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, isolate_session_state=True)
            if self._session_config.HasField('cluster_def'):
                raise ValueError('You cannot provide a ClusterResolver and session_config.cluster_def.')
            if self._cluster_spec:
                self._session_config.cluster_def.CopyFrom(self._cluster_spec.as_cluster_def())

    def _maybe_overwrite_session_config_for_distributed_training(self):
        pass

    @property
    def evaluation_master(self):
        return self._evaluation_master

    @property
    def master(self):
        return self._master

    @property
    def tpu_config(self):
        return self._tpu_config

    @property
    def cluster(self):
        return self._cluster

    def replace(self, **kwargs):
        if 'tpu_config' not in kwargs:
            return super(RunConfig, self).replace(**kwargs)
        tpu_config = kwargs.pop('tpu_config')
        new_instance = super(RunConfig, self).replace(**kwargs)
        new_instance._tpu_config = tpu_config
        return new_instance