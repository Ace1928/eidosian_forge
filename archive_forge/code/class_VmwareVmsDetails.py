from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareVmsDetails(_messages.Message):
    """VmwareVmsDetails describes VMs in vCenter.

  Fields:
    details: The details of the vmware VMs.
  """
    details = _messages.MessageField('VmwareVmDetails', 1, repeated=True)