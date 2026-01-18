from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ReidentifyContentResponse(_messages.Message):
    """Results of re-identifying an item.

  Fields:
    item: The re-identified item.
    overview: An overview of the changes that were made to the `item`.
  """
    item = _messages.MessageField('GooglePrivacyDlpV2ContentItem', 1)
    overview = _messages.MessageField('GooglePrivacyDlpV2TransformationOverview', 2)