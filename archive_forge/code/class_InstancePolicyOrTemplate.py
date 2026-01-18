from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancePolicyOrTemplate(_messages.Message):
    """InstancePolicyOrTemplate lets you define the type of resources to use
  for this job either with an InstancePolicy or an instance template. If
  undefined, Batch picks the type of VM to use and doesn't include optional VM
  resources such as GPUs and extra disks.

  Fields:
    installGpuDrivers: Set this field true if users want Batch to help fetch
      drivers from a third party location and install them for GPUs specified
      in policy.accelerators or instance_template on their behalf. Default is
      false. For Container-Optimized Image cases, Batch will install the
      accelerator driver following milestones of
      https://cloud.google.com/container-optimized-os/docs/release-notes. For
      non Container-Optimized Image cases, following
      https://github.com/GoogleCloudPlatform/compute-gpu-
      installation/blob/main/linux/install_gpu_driver.py.
    instanceTemplate: Name of an instance template used to create VMs. Named
      the field as 'instance_template' instead of 'template' to avoid c++
      keyword conflict.
    policy: InstancePolicy.
  """
    installGpuDrivers = _messages.BooleanField(1)
    instanceTemplate = _messages.StringField(2)
    policy = _messages.MessageField('InstancePolicy', 3)