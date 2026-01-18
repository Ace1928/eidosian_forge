from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ImageRedactionConfig(_messages.Message):
    """Configuration for determining how redaction of images should occur.

  Fields:
    infoType: Only one per info_type should be provided per request. If not
      specified, and redact_all_text is false, the DLP API will redact all
      text that it matches against all info_types that are found, but not
      specified in another ImageRedactionConfig.
    redactAllText: If true, all text found in the image, regardless whether it
      matches an info_type, is redacted. Only one should be provided.
    redactionColor: The color to use when redacting content from an image. If
      not specified, the default is black.
  """
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 1)
    redactAllText = _messages.BooleanField(2)
    redactionColor = _messages.MessageField('GooglePrivacyDlpV2Color', 3)