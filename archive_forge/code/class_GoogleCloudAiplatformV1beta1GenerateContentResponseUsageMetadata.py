from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GenerateContentResponseUsageMetadata(_messages.Message):
    """Usage metadata about response(s).

  Fields:
    candidatesTokenCount: Number of tokens in the response(s).
    promptTokenCount: Number of tokens in the request.
    totalTokenCount: A integer attribute.
  """
    candidatesTokenCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    promptTokenCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    totalTokenCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)