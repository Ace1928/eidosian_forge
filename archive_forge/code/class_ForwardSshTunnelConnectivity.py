from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardSshTunnelConnectivity(_messages.Message):
    """Forward SSH Tunnel connectivity.

  Fields:
    hostname: Required. Hostname for the SSH tunnel.
    password: Input only. SSH password.
    port: Port for the SSH tunnel, default value is 22.
    privateKey: Input only. SSH private key.
    username: Required. Username for the SSH tunnel.
  """
    hostname = _messages.StringField(1)
    password = _messages.StringField(2)
    port = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    privateKey = _messages.StringField(4)
    username = _messages.StringField(5)