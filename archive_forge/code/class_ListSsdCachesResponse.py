from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSsdCachesResponse(_messages.Message):
    """The response for ListSsdCaches.

  Fields:
    nextPageToken: `next_page_token` can be sent in a subsequent ListSsdCaches
      call to fetch more of the matching SSD caches.
    ssdCaches: The list of requested SSD caches.
  """
    nextPageToken = _messages.StringField(1)
    ssdCaches = _messages.MessageField('SsdCache', 2, repeated=True)