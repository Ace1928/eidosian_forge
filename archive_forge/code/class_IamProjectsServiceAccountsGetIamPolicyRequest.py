from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsGetIamPolicyRequest(_messages.Message):
    """A IamProjectsServiceAccountsGetIamPolicyRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy is being requested.
      `resource` is usually specified as a path, such as
      `projects/*project*/zones/*zone*/disks/*disk*`.  The format for the path
      specified in this value is resource specific and is specified in the
      `getIamPolicy` documentation.
  """
    resource = _messages.StringField(1, required=True)