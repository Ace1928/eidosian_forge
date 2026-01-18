from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsConnectionsPatchRequest(_messages.Message):
    """A DlpProjectsLocationsConnectionsPatchRequest object.

  Fields:
    googlePrivacyDlpV2UpdateConnectionRequest: A
      GooglePrivacyDlpV2UpdateConnectionRequest resource to be passed as the
      request body.
    name: Required. Resource name in the format:
      "projects/{project}/locations/{location}/connections/{connection}".
  """
    googlePrivacyDlpV2UpdateConnectionRequest = _messages.MessageField('GooglePrivacyDlpV2UpdateConnectionRequest', 1)
    name = _messages.StringField(2, required=True)