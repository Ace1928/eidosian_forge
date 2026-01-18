from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class DnsManagedZonesListRequest(_messages.Message):
    """A DnsManagedZonesListRequest object.

  Fields:
    dnsName: Restricts the list to return only zones with this domain name.
    maxResults: Optional. Maximum number of results to be returned. If
      unspecified, the server will decide how many results to return.
    pageToken: Optional. A tag returned by a previous list request that was
      truncated. Use this parameter to continue a previous list request.
    project: Identifies the project addressed by this request.
  """
    dnsName = _messages.StringField(1)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    project = _messages.StringField(4, required=True)