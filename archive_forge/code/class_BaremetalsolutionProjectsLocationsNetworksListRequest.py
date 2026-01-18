from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNetworksListRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNetworksListRequest object.

  Fields:
    filter: List filter.
    pageSize: Requested page size. The server might return fewer items than
      requested. If unspecified, server will pick an appropriate default.
    pageToken: A token identifying a page of results from the server.
    parent: Required. Parent value for ListNetworksRequest.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)