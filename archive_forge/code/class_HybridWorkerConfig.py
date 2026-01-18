from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HybridWorkerConfig(_messages.Message):
    """These settings can be applied to a user's build operations. Next ID: 4

  Fields:
    diskSizeGb: The disk size (in GB) which is requested for the build
      container. Defaults to 10 GB.
    memoryGb: The memory (in GB) which is requested for the build container.
      Defaults to 4 GB.
    vcpuCount: The number of vCPUs which are requested for the build
      container. Defaults to 1.
  """
    diskSizeGb = _messages.IntegerField(1)
    memoryGb = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    vcpuCount = _messages.FloatField(3, variant=_messages.Variant.FLOAT)