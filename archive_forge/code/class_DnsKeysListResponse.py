from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsKeysListResponse(_messages.Message):
    """The response to a request to enumerate DnsKeys in a ManagedZone.

  Fields:
    dnsKeys: The requested resources.
    header: A ResponseHeader attribute.
    kind: Type of resource.
    nextPageToken: The presence of this field indicates that there exist more
      results following your last page of results in pagination order. To
      fetch them, make another list request using this value as your
      pagination token. In this way you can retrieve the complete contents of
      even very large collections one page at a time. However, if the contents
      of the collection change between the first and last paginated list
      request, the set of all elements returned are an inconsistent view of
      the collection. There is no way to retrieve a "snapshot" of collections
      larger than the maximum page size.
  """
    dnsKeys = _messages.MessageField('DnsKey', 1, repeated=True)
    header = _messages.MessageField('ResponseHeader', 2)
    kind = _messages.StringField(3, default='dns#dnsKeysListResponse')
    nextPageToken = _messages.StringField(4)