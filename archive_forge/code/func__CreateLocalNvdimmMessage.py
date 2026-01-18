from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def _CreateLocalNvdimmMessage(resources, messages, size_bytes=None, location=None, scope=None, project=None):
    """Create a message representing a local NVDIMM."""
    if location:
        disk_type_ref = instance_utils.ParseDiskType(resources, NVDIMM_DISK_TYPE, project, location, scope)
        disk_type = disk_type_ref.SelfLink()
    else:
        disk_type = NVDIMM_DISK_TYPE
    local_nvdimm = messages.AttachedDisk(type=messages.AttachedDisk.TypeValueValuesEnum.SCRATCH, autoDelete=True, interface=messages.AttachedDisk.InterfaceValueValuesEnum.NVDIMM, mode=messages.AttachedDisk.ModeValueValuesEnum.READ_WRITE, initializeParams=messages.AttachedDiskInitializeParams(diskType=disk_type))
    if size_bytes is not None:
        local_nvdimm.diskSizeGb = utils.BytesToGb(size_bytes)
    return local_nvdimm