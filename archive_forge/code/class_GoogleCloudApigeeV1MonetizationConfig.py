from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1MonetizationConfig(_messages.Message):
    """Configuration for the Monetization add-on.

  Fields:
    enabled: Flag that specifies whether the Monetization add-on is enabled.
    expiresAt: Output only. Time at which the Monetization add-on expires in
      in milliseconds since epoch. If unspecified, the add-on will never
      expire.
  """
    enabled = _messages.BooleanField(1)
    expiresAt = _messages.IntegerField(2)