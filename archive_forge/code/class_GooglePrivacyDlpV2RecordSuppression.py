from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RecordSuppression(_messages.Message):
    """Configuration to suppress records whose suppression conditions evaluate
  to true.

  Fields:
    condition: A condition that when it evaluates to true will result in the
      record being evaluated to be suppressed from the transformed content.
  """
    condition = _messages.MessageField('GooglePrivacyDlpV2RecordCondition', 1)