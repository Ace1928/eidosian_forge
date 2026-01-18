from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsDnsKeysListRequest(_messages.Message):
    """A DnsDnsKeysListRequest object.

  Fields:
    digestType: An optional comma-separated list of digest types to compute
      and display for key signing keys. If omitted, the recommended digest
      type is computed and displayed.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or ID.
    maxResults: Optional. Maximum number of results to be returned. If
      unspecified, the server decides how many results to return.
    pageToken: Optional. A tag returned by a previous list request that was
      truncated. Use this parameter to continue a previous list request.
    project: Identifies the project addressed by this request.
  """
    digestType = _messages.StringField(1)
    managedZone = _messages.StringField(2, required=True)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    project = _messages.StringField(5, required=True)