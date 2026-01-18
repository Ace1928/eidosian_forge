from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestIamPermissionsRequest(_messages.Message):
    """Request message for `TestIamPermissions` method.

  Fields:
    permissions: The set of permissions to check for the `resource`.
      Permissions with wildcards (such as '*' or 'storage.*') are not allowed.
      For more information see [IAM
      Overview](https://cloud.google.com/iam/docs/overview#permissions).
  """
    permissions = _messages.StringField(1, repeated=True)