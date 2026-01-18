from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsTestIamPermissionsRequest(_messages.Message):
    """A IamProjectsServiceAccountsTestIamPermissionsRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy detail is being
      requested. `resource` is usually specified as a path, such as
      `projects/*project*/zones/*zone*/disks/*disk*`.  The format for the path
      specified in this value is resource specific and is specified in the
      `testIamPermissions` documentation.
    testIamPermissionsRequest: A TestIamPermissionsRequest resource to be
      passed as the request body.
  """
    resource = _messages.StringField(1, required=True)
    testIamPermissionsRequest = _messages.MessageField('TestIamPermissionsRequest', 2)