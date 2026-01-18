from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsLogConfig(_messages.Message):
    """Log configuration for the connection.

  Fields:
    enabled: Enabled represents whether logging is enabled or not for a
      connection.
  """
    enabled = _messages.BooleanField(1)