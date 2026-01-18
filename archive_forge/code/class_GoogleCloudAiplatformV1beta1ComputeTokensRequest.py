from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ComputeTokensRequest(_messages.Message):
    """Request message for ComputeTokens RPC call.

  Fields:
    instances: Required. The instances that are the input to token computing
      API call. Schema is identical to the prediction schema of the text
      model, even for the non-text models, like chat models, or Codey models.
  """
    instances = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)