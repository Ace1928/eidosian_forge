from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ContainerPort(_messages.Message):
    """ContainerPort represents a network port in a single container.

  Fields:
    containerPort: Port number the container listens on. This must be a valid
      TCP port number, 0 < container_port < 65536.
    name: If specified, used to specify which protocol to use. Allowed values
      are "http1" and "h2c".
  """
    containerPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    name = _messages.StringField(2)