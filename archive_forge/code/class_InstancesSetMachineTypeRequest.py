from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetMachineTypeRequest(_messages.Message):
    """A InstancesSetMachineTypeRequest object.

  Fields:
    machineType: Full or partial URL of the machine type resource. See Machine
      Types for a full list of machine types. For example: zones/us-
      central1-f/machineTypes/n1-standard-1
  """
    machineType = _messages.StringField(1)