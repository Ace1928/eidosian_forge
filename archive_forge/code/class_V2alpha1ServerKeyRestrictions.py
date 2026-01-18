from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1ServerKeyRestrictions(_messages.Message):
    """Key restrictions that are specific to server keys.

  Fields:
    allowedIps: A list of the caller IP addresses that are allowed when making
      an API call with this key.
  """
    allowedIps = _messages.StringField(1, repeated=True)