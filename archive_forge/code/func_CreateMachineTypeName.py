from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def CreateMachineTypeName(args, confidential_vm_type=None):
    """Create a machine type name for given args and instance reference."""
    machine_type = args.machine_type
    custom_cpu = args.custom_cpu
    custom_memory = args.custom_memory
    vm_type = getattr(args, 'custom_vm_type', None)
    ext = getattr(args, 'custom_extensions', None)
    machine_type_name = InterpretMachineType(machine_type=machine_type, custom_cpu=custom_cpu, custom_memory=custom_memory, ext=ext, vm_type=vm_type, confidential_vm_type=confidential_vm_type)
    return machine_type_name