from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Conditions(_messages.Message):
    """A collection of conditions.

  Fields:
    conditions: A collection of conditions.
  """
    conditions = _messages.MessageField('GooglePrivacyDlpV2Condition', 1, repeated=True)