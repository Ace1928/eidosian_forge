from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UniversalTwoFactor(_messages.Message):
    """Security key information specific to the U2F protocol.

  Fields:
    appId: Application ID for the U2F protocol.
  """
    appId = _messages.StringField(1)