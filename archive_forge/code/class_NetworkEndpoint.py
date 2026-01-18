from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpoint(_messages.Message):
    """A network endpoint over which a TPU worker can be reached.

  Fields:
    accessConfig: The access config for the TPU worker.
    ipAddress: The internal IP address of this network endpoint.
    port: The port of this network endpoint.
  """
    accessConfig = _messages.MessageField('AccessConfig', 1)
    ipAddress = _messages.StringField(2)
    port = _messages.IntegerField(3, variant=_messages.Variant.INT32)