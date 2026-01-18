from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalTestPermissionsRequest(_messages.Message):
    """Request message for `TestPermissions` method.

  Fields:
    permissions: The set of permissions to check for the `resource`.
    resource: Required. The resource for which the permissions are being
      requested.
  """
    permissions = _messages.StringField(1, repeated=True)
    resource = _messages.StringField(2)