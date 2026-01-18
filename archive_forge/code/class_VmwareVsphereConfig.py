from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareVsphereConfig(_messages.Message):
    """VmwareVsphereConfig represents configuration for the VMware VCenter for
  node pool.

  Fields:
    datastore: The name of the vCenter datastore. Inherited from the user
      cluster.
    hostGroups: Vsphere host groups to apply to all VMs in the node pool
    storagePolicyName: The name of the vCenter storage policy. Inherited from
      the user cluster.
    tags: Tags to apply to VMs.
  """
    datastore = _messages.StringField(1)
    hostGroups = _messages.StringField(2, repeated=True)
    storagePolicyName = _messages.StringField(3)
    tags = _messages.MessageField('VmwareVsphereTag', 4, repeated=True)