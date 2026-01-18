from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaRouterApplianceInstance(_messages.Message):
    """A router appliance instance is a Compute Engine virtual machine (VM)
  instance that acts as a BGP speaker. A router appliance instance is
  specified by the URI of the VM and the internal IP address of one of the
  VM's network interfaces.

  Fields:
    ipAddress: The IP address on the VM to use for peering.
    virtualMachine: The URI of the VM.
  """
    ipAddress = _messages.StringField(1)
    virtualMachine = _messages.StringField(2)