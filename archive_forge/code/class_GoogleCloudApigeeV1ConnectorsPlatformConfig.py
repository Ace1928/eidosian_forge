from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ConnectorsPlatformConfig(_messages.Message):
    """Configuration for the Connectors Platform add-on.

  Fields:
    enabled: Flag that specifies whether the Connectors Platform add-on is
      enabled.
    expiresAt: Output only. Time at which the Connectors Platform add-on
      expires in milliseconds since epoch. If unspecified, the add-on will
      never expire.
  """
    enabled = _messages.BooleanField(1)
    expiresAt = _messages.IntegerField(2)