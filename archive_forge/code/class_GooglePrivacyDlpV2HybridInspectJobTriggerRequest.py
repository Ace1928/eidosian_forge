from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2HybridInspectJobTriggerRequest(_messages.Message):
    """Request to search for potentially sensitive info in a custom location.

  Fields:
    hybridItem: The item to inspect.
  """
    hybridItem = _messages.MessageField('GooglePrivacyDlpV2HybridContentItem', 1)