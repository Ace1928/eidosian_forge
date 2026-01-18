from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkaffoldVersion(_messages.Message):
    """Details of a supported Skaffold version.

  Fields:
    maintenanceModeTime: The time at which this version of Skaffold will enter
      maintenance mode.
    supportEndDate: Date when this version is expected to no longer be
      supported.
    supportExpirationTime: The time at which this version of Skaffold will no
      longer be supported.
    version: Release version number. For example, "1.20.3".
  """
    maintenanceModeTime = _messages.StringField(1)
    supportEndDate = _messages.MessageField('Date', 2)
    supportExpirationTime = _messages.StringField(3)
    version = _messages.StringField(4)