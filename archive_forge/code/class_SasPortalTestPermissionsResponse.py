from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalTestPermissionsResponse(_messages.Message):
    """Response message for `TestPermissions` method.

  Fields:
    permissions: A set of permissions that the caller is allowed.
  """
    permissions = _messages.StringField(1, repeated=True)