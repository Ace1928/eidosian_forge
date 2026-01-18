from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestPermissionsRequest(_messages.Message):
    """A TestPermissionsRequest object.

  Fields:
    permissions: The set of permissions to check for the 'resource'.
      Permissions with wildcards (such as '*' or 'storage.*') are not allowed.
  """
    permissions = _messages.StringField(1, repeated=True)