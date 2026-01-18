from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsJobTriggersActivateRequest(_messages.Message):
    """A DlpProjectsJobTriggersActivateRequest object.

  Fields:
    googlePrivacyDlpV2ActivateJobTriggerRequest: A
      GooglePrivacyDlpV2ActivateJobTriggerRequest resource to be passed as the
      request body.
    name: Required. Resource name of the trigger to activate, for example
      `projects/dlp-test-project/jobTriggers/53234423`.
  """
    googlePrivacyDlpV2ActivateJobTriggerRequest = _messages.MessageField('GooglePrivacyDlpV2ActivateJobTriggerRequest', 1)
    name = _messages.StringField(2, required=True)