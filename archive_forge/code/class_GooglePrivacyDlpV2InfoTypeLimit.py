from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeLimit(_messages.Message):
    """Max findings configuration per infoType, per content item or long
  running DlpJob.

  Fields:
    infoType: Type of information the findings limit applies to. Only one
      limit per info_type should be provided. If InfoTypeLimit does not have
      an info_type, the DLP API applies the limit against all info_types that
      are found but not specified in another InfoTypeLimit.
    maxFindings: Max findings limit for the given infoType.
  """
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 1)
    maxFindings = _messages.IntegerField(2, variant=_messages.Variant.INT32)