from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSelector(_messages.Message):
    """Specifies roles and/or permissions to analyze, to determine both the
  identities possessing them and the resources they control. If multiple
  values are specified, results will include roles or permissions matching any
  of them. The total number of roles and permissions should be equal or less
  than 10.

  Fields:
    permissions: Optional. The permissions to appear in result.
    roles: Optional. The roles to appear in result.
  """
    permissions = _messages.StringField(1, repeated=True)
    roles = _messages.StringField(2, repeated=True)