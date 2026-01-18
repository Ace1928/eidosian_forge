from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListHotTabletsResponse(_messages.Message):
    """Response message for BigtableInstanceAdmin.ListHotTablets.

  Fields:
    hotTablets: List of hot tablets in the tables of the requested cluster
      that fall within the requested time range. Hot tablets are ordered by
      node cpu usage percent. If there are multiple hot tablets that
      correspond to the same tablet within a 15-minute interval, only the hot
      tablet with the highest node cpu usage will be included in the response.
    nextPageToken: Set if not all hot tablets could be returned in a single
      response. Pass this value to `page_token` in another request to get the
      next page of results.
  """
    hotTablets = _messages.MessageField('HotTablet', 1, repeated=True)
    nextPageToken = _messages.StringField(2)