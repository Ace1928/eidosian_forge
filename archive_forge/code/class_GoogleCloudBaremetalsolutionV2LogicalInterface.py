from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBaremetalsolutionV2LogicalInterface(_messages.Message):
    """Each logical interface represents a logical abstraction of the
  underlying physical interface (for eg. bond, nic) of the instance. Each
  logical interface can effectively map to multiple network-IP pairs and still
  be mapped to one underlying physical interface.

  Fields:
    interfaceIndex: The index of the logical interface mapping to the index of
      the hardware bond or nic on the chosen network template. This field is
      deprecated.
    logicalNetworkInterfaces: List of logical network interfaces within a
      logical interface.
    name: Interface name. This is of syntax or and forms part of the network
      template name.
  """
    interfaceIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    logicalNetworkInterfaces = _messages.MessageField('LogicalNetworkInterface', 2, repeated=True)
    name = _messages.StringField(3)