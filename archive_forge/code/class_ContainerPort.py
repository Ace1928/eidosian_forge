from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerPort(_messages.Message):
    """ContainerPort represents a network port in a single container.

  Fields:
    containerPort: Port number the container listens on. If present, this must
      be a valid port number, 0 < x < 65536. If not present, it will default
      to port 8080. For more information, see
      https://cloud.google.com/run/docs/container-contract#port
    name: If specified, used to specify which protocol to use. Allowed values
      are "http1" and "h2c".
    protocol: Protocol for port. Must be "TCP". Defaults to "TCP".
  """
    containerPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    name = _messages.StringField(2)
    protocol = _messages.StringField(3)