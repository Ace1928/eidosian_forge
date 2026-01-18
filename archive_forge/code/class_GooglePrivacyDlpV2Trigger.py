from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Trigger(_messages.Message):
    """What event needs to occur for a new job to be started.

  Fields:
    manual: For use with hybrid jobs. Jobs must be manually created and
      finished.
    schedule: Create a job on a repeating basis based on the elapse of time.
  """
    manual = _messages.MessageField('GooglePrivacyDlpV2Manual', 1)
    schedule = _messages.MessageField('GooglePrivacyDlpV2Schedule', 2)