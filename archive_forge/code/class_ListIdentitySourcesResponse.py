from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListIdentitySourcesResponse(_messages.Message):
    """Response message for VmwareEngine.ListIdentitySources

  Fields:
    identitySources: A list of private cloud identity sources.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Resources that could not be reached when making an aggregated
      query using wildcards.
  """
    identitySources = _messages.MessageField('IdentitySource', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)