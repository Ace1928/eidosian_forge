from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsRegionalEndpointsListRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsRegionalEndpointsListRequest
  object.

  Fields:
    filter: A filter expression that filters the results listed in the
      response.
    orderBy: Sort the results by a certain order.
    pageSize: Requested page size. Server may return fewer items than
      requested. If unspecified, server will pick an appropriate default.
    pageToken: A page token.
    parent: Required. The parent resource's name of the RegionalEndpoint.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)