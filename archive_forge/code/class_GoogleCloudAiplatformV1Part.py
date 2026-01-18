from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Part(_messages.Message):
    """A datatype containing media that is part of a multi-part `Content`
  message. A `Part` consists of data which has an associated datatype. A
  `Part` can only contain one of the accepted types in `Part.data`. A `Part`
  must have a fixed IANA MIME type identifying the type and subtype of the
  media if `inline_data` or `file_data` field is filled with raw bytes.

  Fields:
    fileData: Optional. URI based data.
    functionCall: Optional. A predicted [FunctionCall] returned from the model
      that contains a string representing the [FunctionDeclaration.name] with
      the parameters and their values.
    functionResponse: Optional. The result output of a [FunctionCall] that
      contains a string representing the [FunctionDeclaration.name] and a
      structured JSON object containing any output from the function call. It
      is used as context to the model.
    inlineData: Optional. Inlined bytes data.
    text: Optional. Text part (can be code).
    videoMetadata: Optional. Video metadata. The metadata should only be
      specified while the video data is presented in inline_data or file_data.
  """
    fileData = _messages.MessageField('GoogleCloudAiplatformV1FileData', 1)
    functionCall = _messages.MessageField('GoogleCloudAiplatformV1FunctionCall', 2)
    functionResponse = _messages.MessageField('GoogleCloudAiplatformV1FunctionResponse', 3)
    inlineData = _messages.MessageField('GoogleCloudAiplatformV1Blob', 4)
    text = _messages.StringField(5)
    videoMetadata = _messages.MessageField('GoogleCloudAiplatformV1VideoMetadata', 6)