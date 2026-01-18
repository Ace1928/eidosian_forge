from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsPatchRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsPatchRequest
  object.

  Fields:
    authorizedOrgsDesc: A AuthorizedOrgsDesc resource to be passed as the
      request body.
    name: Resource name for the `AuthorizedOrgsDesc`. Format: `accessPolicies/
      {access_policy}/authorizedOrgsDescs/{authorized_orgs_desc}`. The
      `authorized_orgs_desc` component must begin with a letter, followed by
      alphanumeric characters or `_`. After you create an
      `AuthorizedOrgsDesc`, you cannot change its `name`.
    updateMask: Required. Mask to control which fields get updated. Must be
      non-empty.
  """
    authorizedOrgsDesc = _messages.MessageField('AuthorizedOrgsDesc', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)