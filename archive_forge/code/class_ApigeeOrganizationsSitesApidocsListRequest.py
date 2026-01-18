from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApidocsListRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApidocsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of items to return. The service may
      return fewer than this value. If unspecified, at most 25 books will be
      returned. The maximum value is 100; values above 100 will be coerced to
      100.
    pageToken: Optional. A page token, received from a previous `ListApiDocs`
      call. Provide this to retrieve the subsequent page.
    parent: Required. Name of the portal. Use the following structure in your
      request: `organizations/{org}/sites/{site}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)