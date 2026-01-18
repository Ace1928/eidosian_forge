from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceSelectionResult(_messages.Message):
    """Defines a mapping from machine types to the number of VMs that are
  created with each machine type.

  Fields:
    machineType: Output only. Full machine-type names, e.g. "n1-standard-16".
    vmCount: Output only. Number of VM provisioned with the machine_type.
  """
    machineType = _messages.StringField(1)
    vmCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)