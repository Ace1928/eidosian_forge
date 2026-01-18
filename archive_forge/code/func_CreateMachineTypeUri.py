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
def CreateMachineTypeUri(args, compute_client, resource_parser, project, location, scope, confidential_vm_type=False):
    """Create a machine type URI for given args and instance reference."""
    machine_type = args.machine_type
    custom_cpu = args.custom_cpu
    custom_memory = args.custom_memory
    vm_type = getattr(args, 'custom_vm_type', None)
    ext = getattr(args, 'custom_extensions', None)
    machine_type_name = instance_utils.InterpretMachineType(machine_type=machine_type, custom_cpu=custom_cpu, custom_memory=custom_memory, ext=ext, vm_type=vm_type, confidential_vm_type=confidential_vm_type)
    instance_utils.CheckCustomCpuRamRatio(compute_client, project, location, machine_type_name)
    machine_type_uri = instance_utils.ParseMachineType(resource_parser, machine_type_name, project, location, scope)
    return machine_type_uri