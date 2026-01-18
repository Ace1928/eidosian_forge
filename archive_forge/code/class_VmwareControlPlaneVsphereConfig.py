from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareControlPlaneVsphereConfig(_messages.Message):
    """Specifies control plane node config.

  Fields:
    datastore: The Vsphere datastore used by the control plane Node.
    storagePolicyName: The Vsphere storage policy used by the control plane
      Node.
  """
    datastore = _messages.StringField(1)
    storagePolicyName = _messages.StringField(2)