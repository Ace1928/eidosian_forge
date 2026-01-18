from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpOrganizationsStoredInfoTypesPatchRequest(_messages.Message):
    """A DlpOrganizationsStoredInfoTypesPatchRequest object.

  Fields:
    googlePrivacyDlpV2UpdateStoredInfoTypeRequest: A
      GooglePrivacyDlpV2UpdateStoredInfoTypeRequest resource to be passed as
      the request body.
    name: Required. Resource name of organization and storedInfoType to be
      updated, for example `organizations/433245324/storedInfoTypes/432452342`
      or projects/project-id/storedInfoTypes/432452342.
  """
    googlePrivacyDlpV2UpdateStoredInfoTypeRequest = _messages.MessageField('GooglePrivacyDlpV2UpdateStoredInfoTypeRequest', 1)
    name = _messages.StringField(2, required=True)