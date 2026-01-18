from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1BuildOptionsPoolOptionWorkerConfig(_messages.Message):
    """Configuration per workload for both Private Pools and Hybrid Pools.

  Fields:
    diskSizeGb: The disk size (in GB) which is requested for the build
      container. If unset, a value of 10 GB will be used.
    memoryGb: The memory (in GB) which is requested for the build container.
      If unset, a value of 4 GB will be used.
    vcpuCount: The number of vCPUs which are requested for the build
      container. If unset, a value of 1 will be used.
  """
    diskSizeGb = _messages.IntegerField(1)
    memoryGb = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    vcpuCount = _messages.FloatField(3, variant=_messages.Variant.FLOAT)