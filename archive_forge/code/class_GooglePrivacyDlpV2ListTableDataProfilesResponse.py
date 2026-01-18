from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListTableDataProfilesResponse(_messages.Message):
    """List of profiles generated for a given organization or project.

  Fields:
    nextPageToken: The next page token.
    tableDataProfiles: List of data profiles.
  """
    nextPageToken = _messages.StringField(1)
    tableDataProfiles = _messages.MessageField('GooglePrivacyDlpV2TableDataProfile', 2, repeated=True)