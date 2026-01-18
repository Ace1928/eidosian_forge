from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeNodePoolAcceleratorConfig(_messages.Message):
    """A GkeNodeConfigAcceleratorConfig represents a Hardware Accelerator
  request for a node pool.

  Fields:
    acceleratorCount: The number of accelerator cards exposed to an instance.
    acceleratorType: The accelerator type resource namename (see GPUs on
      Compute Engine).
    gpuPartitionSize: Size of partitions to create on the GPU. Valid values
      are described in the NVIDIA mig user guide
      (https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#partitioning).
  """
    acceleratorCount = _messages.IntegerField(1)
    acceleratorType = _messages.StringField(2)
    gpuPartitionSize = _messages.StringField(3)