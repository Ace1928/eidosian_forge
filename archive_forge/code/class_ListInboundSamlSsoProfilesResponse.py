from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInboundSamlSsoProfilesResponse(_messages.Message):
    """Response of the InboundSamlSsoProfilesService.ListInboundSamlSsoProfiles
  method.

  Fields:
    inboundSamlSsoProfiles: List of InboundSamlSsoProfiles.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    inboundSamlSsoProfiles = _messages.MessageField('InboundSamlSsoProfile', 1, repeated=True)
    nextPageToken = _messages.StringField(2)