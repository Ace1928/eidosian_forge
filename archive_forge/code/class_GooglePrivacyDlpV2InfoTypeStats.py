from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeStats(_messages.Message):
    """Statistics regarding a specific InfoType.

  Fields:
    count: Number of findings for this infoType.
    infoType: The type of finding this stat is for.
  """
    count = _messages.IntegerField(1)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 2)