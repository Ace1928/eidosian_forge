from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolNameMatchInput(_messages.Message):
    """Input for tool name match metric.

  Fields:
    instances: Required. Repeated tool name match instances.
    metricSpec: Required. Spec for tool name match metric.
  """
    instances = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolNameMatchInstance', 1, repeated=True)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolNameMatchSpec', 2)