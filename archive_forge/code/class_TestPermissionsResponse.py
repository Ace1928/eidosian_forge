from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestPermissionsResponse(_messages.Message):
    """A TestPermissionsResponse object.

  Fields:
    permissions: A subset of `TestPermissionsRequest.permissions` that the
      caller is allowed.
  """
    permissions = _messages.StringField(1, repeated=True)