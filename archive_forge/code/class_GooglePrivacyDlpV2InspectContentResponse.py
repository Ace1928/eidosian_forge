from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectContentResponse(_messages.Message):
    """Results of inspecting an item.

  Fields:
    result: The findings.
  """
    result = _messages.MessageField('GooglePrivacyDlpV2InspectResult', 1)