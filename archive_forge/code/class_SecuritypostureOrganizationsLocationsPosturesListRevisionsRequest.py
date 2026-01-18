from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPosturesListRevisionsRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPosturesListRevisionsRequest
  object.

  Fields:
    name: Required. Name value for ListPostureRevisionsRequest.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick 100 as default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)