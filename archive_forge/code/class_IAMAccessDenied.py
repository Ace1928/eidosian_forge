from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IAMAccessDenied(_messages.Message):
    """PAM's service account is being denied access by Cloud IAM. This can be
  fixed by granting a role that contains the missing permissions to the
  service account or exempting it from deny policies if they are blocking the
  access.

  Fields:
    missingPermissions: List of permissions that are being denied.
  """
    missingPermissions = _messages.StringField(1, repeated=True)