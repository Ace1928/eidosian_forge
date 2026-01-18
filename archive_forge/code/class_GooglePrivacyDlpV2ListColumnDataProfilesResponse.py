from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListColumnDataProfilesResponse(_messages.Message):
    """List of profiles generated for a given organization or project.

  Fields:
    columnDataProfiles: List of data profiles.
    nextPageToken: The next page token.
  """
    columnDataProfiles = _messages.MessageField('GooglePrivacyDlpV2ColumnDataProfile', 1, repeated=True)
    nextPageToken = _messages.StringField(2)