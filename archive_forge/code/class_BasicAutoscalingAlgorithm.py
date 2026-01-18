from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BasicAutoscalingAlgorithm(_messages.Message):
    """Basic algorithm for autoscaling.

  Fields:
    cooldownPeriod: Optional. Duration between scaling events. A scaling
      period starts after the update operation from the previous event has
      completed.Bounds: 2m, 1d. Default: 2m.
    sparkStandaloneConfig: Optional. Spark Standalone autoscaling
      configuration
    yarnConfig: Optional. YARN autoscaling configuration.
  """
    cooldownPeriod = _messages.StringField(1)
    sparkStandaloneConfig = _messages.MessageField('SparkStandaloneAutoscalingConfig', 2)
    yarnConfig = _messages.MessageField('BasicYarnAutoscalingConfig', 3)