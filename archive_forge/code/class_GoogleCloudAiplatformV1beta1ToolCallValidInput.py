from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolCallValidInput(_messages.Message):
    """Input for tool call valid metric.

  Fields:
    instances: Required. Repeated tool call valid instances.
    metricSpec: Required. Spec for tool call valid metric.
  """
    instances = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolCallValidInstance', 1, repeated=True)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolCallValidSpec', 2)