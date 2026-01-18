from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretManagerConfig(_messages.Message):
    """SecretManagerConfig is config for secret manager enablement.

  Fields:
    enabled: Enable/Disable Secret Manager Config.
  """
    enabled = _messages.BooleanField(1)