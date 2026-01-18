from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsResponsePoliciesListRequest(_messages.Message):
    """A DnsResponsePoliciesListRequest object.

  Fields:
    maxResults: Optional. Maximum number of results to be returned. If
      unspecified, the server decides how many results to return.
    pageToken: Optional. A tag returned by a previous list request that was
      truncated. Use this parameter to continue a previous list request.
    project: Identifies the project addressed by this request.
  """
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    project = _messages.StringField(3, required=True)