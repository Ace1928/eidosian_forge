from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsAutoscalingGroupMetricsCollection(_messages.Message):
    """Configuration related to CloudWatch metrics collection in an AWS Auto
  Scaling group.

  Fields:
    granularity: Required. The frequency at which EC2 Auto Scaling sends
      aggregated data to AWS CloudWatch. The only valid value is "1Minute".
    metrics: Optional. The metrics to enable. For a list of valid metrics, see
      https://docs.aws.amazon.com/autoscaling/ec2/APIReference/API_EnableMetri
      csCollection.html. If you specify Granularity and don't specify any
      metrics, all metrics are enabled.
  """
    granularity = _messages.StringField(1)
    metrics = _messages.StringField(2, repeated=True)