from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityProfilesV2Response(_messages.Message):
    """Response for ListSecurityProfilesV2.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    securityProfilesV2: List of security profiles in the organization.
  """
    nextPageToken = _messages.StringField(1)
    securityProfilesV2 = _messages.MessageField('GoogleCloudApigeeV1SecurityProfileV2', 2, repeated=True)