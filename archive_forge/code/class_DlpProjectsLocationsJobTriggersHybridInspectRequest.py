from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsJobTriggersHybridInspectRequest(_messages.Message):
    """A DlpProjectsLocationsJobTriggersHybridInspectRequest object.

  Fields:
    googlePrivacyDlpV2HybridInspectJobTriggerRequest: A
      GooglePrivacyDlpV2HybridInspectJobTriggerRequest resource to be passed
      as the request body.
    name: Required. Resource name of the trigger to execute a hybrid inspect
      on, for example `projects/dlp-test-project/jobTriggers/53234423`.
  """
    googlePrivacyDlpV2HybridInspectJobTriggerRequest = _messages.MessageField('GooglePrivacyDlpV2HybridInspectJobTriggerRequest', 1)
    name = _messages.StringField(2, required=True)