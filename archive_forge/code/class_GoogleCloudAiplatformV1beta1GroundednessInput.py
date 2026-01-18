from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GroundednessInput(_messages.Message):
    """Input for groundedness metric.

  Fields:
    instance: Required. Groundedness instance.
    metricSpec: Required. Spec for groundedness metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundednessInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundednessSpec', 2)