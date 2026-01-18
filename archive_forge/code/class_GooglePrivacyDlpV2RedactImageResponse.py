from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RedactImageResponse(_messages.Message):
    """Results of redacting an image.

  Fields:
    extractedText: If an image was being inspected and the InspectConfig's
      include_quote was set to true, then this field will include all text, if
      any, that was found in the image.
    inspectResult: The findings. Populated when include_findings in the
      request is true.
    redactedImage: The redacted image. The type will be the same as the
      original image.
  """
    extractedText = _messages.StringField(1)
    inspectResult = _messages.MessageField('GooglePrivacyDlpV2InspectResult', 2)
    redactedImage = _messages.BytesField(3)