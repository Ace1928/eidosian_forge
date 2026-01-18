from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ImageTransformation(_messages.Message):
    """Configuration for determining how redaction of images should occur.

  Fields:
    allInfoTypes: Apply transformation to all findings not specified in other
      ImageTransformation's selected_info_types. Only one instance is allowed
      within the ImageTransformations message.
    allText: Apply transformation to all text that doesn't match an infoType.
      Only one instance is allowed within the ImageTransformations message.
    redactionColor: The color to use when redacting content from an image. If
      not specified, the default is black.
    selectedInfoTypes: Apply transformation to the selected info_types.
  """
    allInfoTypes = _messages.MessageField('GooglePrivacyDlpV2AllInfoTypes', 1)
    allText = _messages.MessageField('GooglePrivacyDlpV2AllText', 2)
    redactionColor = _messages.MessageField('GooglePrivacyDlpV2Color', 3)
    selectedInfoTypes = _messages.MessageField('GooglePrivacyDlpV2SelectedInfoTypes', 4)