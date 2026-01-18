from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareVCenterConfig(_messages.Message):
    """Represents configuration for the VMware VCenter for the user cluster.

  Fields:
    address: Output only. The vCenter IP address.
    caCertData: Contains the vCenter CA certificate public key for SSL
      verification.
    cluster: The name of the vCenter cluster for the user cluster.
    datacenter: The name of the vCenter datacenter for the user cluster.
    datastore: The name of the vCenter datastore for the user cluster.
    folder: The name of the vCenter folder for the user cluster.
    resourcePool: The name of the vCenter resource pool for the user cluster.
    storagePolicyName: The name of the vCenter storage policy for the user
      cluster.
  """
    address = _messages.StringField(1)
    caCertData = _messages.StringField(2)
    cluster = _messages.StringField(3)
    datacenter = _messages.StringField(4)
    datastore = _messages.StringField(5)
    folder = _messages.StringField(6)
    resourcePool = _messages.StringField(7)
    storagePolicyName = _messages.StringField(8)