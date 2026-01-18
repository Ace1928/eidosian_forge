from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPostureDeploymentsListRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPostureDeploymentsListRequest
  object.

  Fields:
    filter: Optional. Filter to be applied on the resource, defined by EBNF
      grammar https://google.aip.dev/assets/misc/ebnf-filtering.txt.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. Parent value for ListPostureDeploymentsRequest.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)