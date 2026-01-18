from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientConnectionConfig(_messages.Message):
    """Client connection configuration

  Fields:
    requireConnectors: Optional. Configuration to enforce connectors only (ex:
      AuthProxy) connections to the database.
    sslConfig: Optional. SSL config option for this instance.
  """
    requireConnectors = _messages.BooleanField(1)
    sslConfig = _messages.MessageField('SslConfig', 2)