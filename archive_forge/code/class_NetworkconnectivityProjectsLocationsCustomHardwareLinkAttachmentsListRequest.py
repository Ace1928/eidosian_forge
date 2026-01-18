from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsListRequest(_messages.Message):
    """A
  NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsListRequest
  object.

  Fields:
    filter: Optional. A filter expression that filters the results listed in
      the response.
    orderBy: Optional. Sort the results by a certain order.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. The parent resource's name of the
      CustomHardwareLinkAttachment.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)