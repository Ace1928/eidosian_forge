from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectResult(_messages.Message):
    """All the findings for a single scanned item.

  Fields:
    findings: List of findings for an item.
    findingsTruncated: If true, then this item might have more findings than
      were returned, and the findings returned are an arbitrary subset of all
      findings. The findings list might be truncated because the input items
      were too large, or because the server reached the maximum amount of
      resources allowed for a single API call. For best results, divide the
      input into smaller batches.
  """
    findings = _messages.MessageField('GooglePrivacyDlpV2Finding', 1, repeated=True)
    findingsTruncated = _messages.BooleanField(2)