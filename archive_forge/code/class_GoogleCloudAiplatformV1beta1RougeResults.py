from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RougeResults(_messages.Message):
    """Results for rouge metric.

  Fields:
    rougeMetricValues: Output only. Rouge metric values.
  """
    rougeMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1RougeMetricValue', 1, repeated=True)