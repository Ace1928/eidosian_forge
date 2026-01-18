from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketMirroringMirroredResourceInfo(_messages.Message):
    """A PacketMirroringMirroredResourceInfo object.

  Fields:
    instances: A set of virtual machine instances that are being mirrored.
      They must live in zones contained in the same region as this
      packetMirroring. Note that this config will apply only to those network
      interfaces of the Instances that belong to the network specified in this
      packetMirroring. You may specify a maximum of 50 Instances.
    subnetworks: A set of subnetworks for which traffic from/to all VM
      instances will be mirrored. They must live in the same region as this
      packetMirroring. You may specify a maximum of 5 subnetworks.
    tags: A set of mirrored tags. Traffic from/to all VM instances that have
      one or more of these tags will be mirrored.
  """
    instances = _messages.MessageField('PacketMirroringMirroredResourceInfoInstanceInfo', 1, repeated=True)
    subnetworks = _messages.MessageField('PacketMirroringMirroredResourceInfoSubnetInfo', 2, repeated=True)
    tags = _messages.StringField(3, repeated=True)