from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityProfileRevisionsResponse(_messages.Message):
    """Response for ListSecurityProfileRevisions.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    securityProfiles: List of security profile revisions. The revisions may be
      attached or unattached to any environment.
  """
    nextPageToken = _messages.StringField(1)
    securityProfiles = _messages.MessageField('GoogleCloudApigeeV1SecurityProfile', 2, repeated=True)