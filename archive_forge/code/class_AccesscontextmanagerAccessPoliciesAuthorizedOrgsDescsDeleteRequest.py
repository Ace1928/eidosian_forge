from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsDeleteRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsDeleteRequest
  object.

  Fields:
    name: Required. Resource name for the Authorized Orgs Desc. Format:
      `accessPolicies/{policy_id}/authorizedOrgsDesc/{authorized_orgs_desc_id}
      `
  """
    name = _messages.StringField(1, required=True)