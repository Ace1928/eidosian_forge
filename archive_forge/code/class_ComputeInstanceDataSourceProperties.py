from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceDataSourceProperties(_messages.Message):
    """ComputeInstanceDataSourceProperties represents the properties of a
  ComputeEngine resource that are stored in the DataSource.

  Fields:
    description: The description of the Compute Engine instance.
    machineType: The machine type of the instance.
    name: Name of the compute instance backed up by the datasource.
    totalDiskCount: The total number of disks attached to the Instance.
    totalDiskSizeGb: The sum of all the disk sizes.
  """
    description = _messages.StringField(1)
    machineType = _messages.StringField(2)
    name = _messages.StringField(3)
    totalDiskCount = _messages.IntegerField(4)
    totalDiskSizeGb = _messages.IntegerField(5)