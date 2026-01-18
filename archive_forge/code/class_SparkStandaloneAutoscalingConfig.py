from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkStandaloneAutoscalingConfig(_messages.Message):
    """Basic autoscaling configurations for Spark Standalone.

  Fields:
    gracefulDecommissionTimeout: Required. Timeout for Spark graceful
      decommissioning of spark workers. Specifies the duration to wait for
      spark worker to complete spark decommissioning tasks before forcefully
      removing workers. Only applicable to downscaling operations.Bounds: 0s,
      1d.
    removeOnlyIdleWorkers: Optional. Remove only idle workers when scaling
      down cluster
    scaleDownFactor: Required. Fraction of required executors to remove from
      Spark Serverless clusters. A scale-down factor of 1.0 will result in
      scaling down so that there are no more executors for the Spark Job.(more
      aggressive scaling). A scale-down factor closer to 0 will result in a
      smaller magnitude of scaling donw (less aggressive scaling).Bounds: 0.0,
      1.0.
    scaleDownMinWorkerFraction: Optional. Minimum scale-down threshold as a
      fraction of total cluster size before scaling occurs. For example, in a
      20-worker cluster, a threshold of 0.1 means the autoscaler must
      recommend at least a 2 worker scale-down for the cluster to scale. A
      threshold of 0 means the autoscaler will scale down on any recommended
      change.Bounds: 0.0, 1.0. Default: 0.0.
    scaleUpFactor: Required. Fraction of required workers to add to Spark
      Standalone clusters. A scale-up factor of 1.0 will result in scaling up
      so that there are no more required workers for the Spark Job (more
      aggressive scaling). A scale-up factor closer to 0 will result in a
      smaller magnitude of scaling up (less aggressive scaling).Bounds: 0.0,
      1.0.
    scaleUpMinWorkerFraction: Optional. Minimum scale-up threshold as a
      fraction of total cluster size before scaling occurs. For example, in a
      20-worker cluster, a threshold of 0.1 means the autoscaler must
      recommend at least a 2-worker scale-up for the cluster to scale. A
      threshold of 0 means the autoscaler will scale up on any recommended
      change.Bounds: 0.0, 1.0. Default: 0.0.
  """
    gracefulDecommissionTimeout = _messages.StringField(1)
    removeOnlyIdleWorkers = _messages.BooleanField(2)
    scaleDownFactor = _messages.FloatField(3)
    scaleDownMinWorkerFraction = _messages.FloatField(4)
    scaleUpFactor = _messages.FloatField(5)
    scaleUpMinWorkerFraction = _messages.FloatField(6)