from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceConnectionMapsResponse(_messages.Message):
    """Response for ListServiceConnectionMaps.

  Fields:
    nextPageToken: The next pagination token in the List response. It should
      be used as page_token for the following request. An empty value means no
      more result.
    serviceConnectionMaps: ServiceConnectionMaps to be returned.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    serviceConnectionMaps = _messages.MessageField('ServiceConnectionMap', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)