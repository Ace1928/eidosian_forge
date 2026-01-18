from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionMediaGenerateContentResponse(_messages.Message):
    """Generate media content response

  Fields:
    response: Response to the user's request.
  """
    response = _messages.MessageField('CloudAiNlLlmProtoServiceGenerateMultiModalResponse', 1)