from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Location(_messages.Message):
    """Specifies the location of the finding.

  Fields:
    byteRange: Zero-based byte offsets delimiting the finding. These are
      relative to the finding's containing element. Note that when the content
      is not textual, this references the UTF-8 encoded textual representation
      of the content. Omitted if content is an image.
    codepointRange: Unicode character offsets delimiting the finding. These
      are relative to the finding's containing element. Provided when the
      content is text.
    container: Information about the container where this finding occurred, if
      available.
    contentLocations: List of nested objects pointing to the precise location
      of the finding within the file or record.
  """
    byteRange = _messages.MessageField('GooglePrivacyDlpV2Range', 1)
    codepointRange = _messages.MessageField('GooglePrivacyDlpV2Range', 2)
    container = _messages.MessageField('GooglePrivacyDlpV2Container', 3)
    contentLocations = _messages.MessageField('GooglePrivacyDlpV2ContentLocation', 4, repeated=True)