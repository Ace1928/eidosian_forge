from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolParameterKVMatchInput(_messages.Message):
    """Input for tool parameter key value match metric.

  Fields:
    instances: Required. Repeated tool parameter key value match instances.
    metricSpec: Required. Spec for tool parameter key value match metric.
  """
    instances = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKVMatchInstance', 1, repeated=True)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKVMatchSpec', 2)