from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CanaryEvaluationMetricLabels(_messages.Message):
    """Labels that can be used to filter Apigee metrics.

  Fields:
    env: The environment ID associated with the metrics.
    instance_id: Required. The instance ID associated with the metrics. In
      Apigee Hybrid, the value is configured during installation.
    location: Required. The location associated with the metrics.
  """
    env = _messages.StringField(1)
    instance_id = _messages.StringField(2)
    location = _messages.StringField(3)