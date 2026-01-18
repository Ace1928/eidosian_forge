from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDlpJobsFinishRequest(_messages.Message):
    """A DlpProjectsLocationsDlpJobsFinishRequest object.

  Fields:
    googlePrivacyDlpV2FinishDlpJobRequest: A
      GooglePrivacyDlpV2FinishDlpJobRequest resource to be passed as the
      request body.
    name: Required. The name of the DlpJob resource to be finished.
  """
    googlePrivacyDlpV2FinishDlpJobRequest = _messages.MessageField('GooglePrivacyDlpV2FinishDlpJobRequest', 1)
    name = _messages.StringField(2, required=True)