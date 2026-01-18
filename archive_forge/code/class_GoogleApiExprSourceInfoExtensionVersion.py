from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprSourceInfoExtensionVersion(_messages.Message):
    """Version

  Fields:
    major: Major version changes indicate different required support level
      from the required components.
    minor: Minor version changes must not change the observed behavior from
      existing implementations, but may be provided informationally.
  """
    major = _messages.IntegerField(1)
    minor = _messages.IntegerField(2)