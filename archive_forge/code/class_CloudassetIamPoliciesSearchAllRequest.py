from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudassetIamPoliciesSearchAllRequest(_messages.Message):
    """A CloudassetIamPoliciesSearchAllRequest object.

  Fields:
    pageSize: Optional. The page size for search result pagination. Page size
      is capped at 500 even if a larger value is given. If set to zero, server
      will pick an appropriate default. Returned results may be fewer than
      requested. When this happens, there could be more results as long as
      `next_page_token` is returned.
    pageToken: Optional. If present, retrieve the next batch of results from
      the preceding call to this method. `page_token` must be the value of
      `next_page_token` from the previous response. The values of all other
      method parameters must be identical to those in the previous call.
    query: Optional. The query statement. Examples: *
      "policy:myuser@mydomain.com" * "policy:(myuser@mydomain.com viewer)"
    scope: Required. The relative name of an asset. The search is limited to
      the resources within the `scope`. The allowed value must be: *
      Organization number (such as "organizations/123") * Folder number(such
      as "folders/1234") * Project number (such as "projects/12345") * Project
      id (such as "projects/abc")
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    query = _messages.StringField(3)
    scope = _messages.StringField(4, required=True)