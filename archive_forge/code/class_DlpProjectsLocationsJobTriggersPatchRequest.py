from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsJobTriggersPatchRequest(_messages.Message):
    """A DlpProjectsLocationsJobTriggersPatchRequest object.

  Fields:
    googlePrivacyDlpV2UpdateJobTriggerRequest: A
      GooglePrivacyDlpV2UpdateJobTriggerRequest resource to be passed as the
      request body.
    name: Required. Resource name of the project and the triggeredJob, for
      example `projects/dlp-test-project/jobTriggers/53234423`.
  """
    googlePrivacyDlpV2UpdateJobTriggerRequest = _messages.MessageField('GooglePrivacyDlpV2UpdateJobTriggerRequest', 1)
    name = _messages.StringField(2, required=True)