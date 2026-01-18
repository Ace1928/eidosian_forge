from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeidentifyContentResponse(_messages.Message):
    """Results of de-identifying a ContentItem.

  Fields:
    item: The de-identified item.
    overview: An overview of the changes that were made on the `item`.
  """
    item = _messages.MessageField('GooglePrivacyDlpV2ContentItem', 1)
    overview = _messages.MessageField('GooglePrivacyDlpV2TransformationOverview', 2)