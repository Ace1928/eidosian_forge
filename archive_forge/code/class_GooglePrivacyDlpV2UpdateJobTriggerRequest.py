from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UpdateJobTriggerRequest(_messages.Message):
    """Request message for UpdateJobTrigger.

  Fields:
    jobTrigger: New JobTrigger value.
    updateMask: Mask to control which fields get updated.
  """
    jobTrigger = _messages.MessageField('GooglePrivacyDlpV2JobTrigger', 1)
    updateMask = _messages.StringField(2)