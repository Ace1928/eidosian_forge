from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsListRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsListRequest
  object.

  Fields:
    pageSize: Number of Authorized Orgs Descs to include in the list. Default
      100.
    pageToken: Next page token for the next batch of Authorized Orgs Desc
      instances. Defaults to the first page of results.
    parent: Required. Resource name for the access policy to list Authorized
      Orgs Desc from. Format: `accessPolicies/{policy_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)