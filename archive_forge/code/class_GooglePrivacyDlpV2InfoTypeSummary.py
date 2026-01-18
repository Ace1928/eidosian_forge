from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeSummary(_messages.Message):
    """The infoType details for this column.

  Fields:
    estimatedPrevalence: Not populated for predicted infotypes.
    infoType: The infoType.
  """
    estimatedPrevalence = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 2)