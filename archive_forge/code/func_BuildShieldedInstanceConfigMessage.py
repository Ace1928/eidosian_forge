from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instance_templates import mesh_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def BuildShieldedInstanceConfigMessage(messages, args):
    """Common routine for creating instance template.

  Build a shielded VM config message.

  Args:
      messages: The client messages.
      args: the arguments passed to the test.

  Returns:
      A shielded VM config message.
  """
    shielded_instance_config_message = None
    enable_secure_boot = None
    enable_vtpm = None
    enable_integrity_monitoring = None
    if not (hasattr(args, 'shielded_vm_secure_boot') or hasattr(args, 'shielded_vm_vtpm') or hasattr(args, 'shielded_vm_integrity_monitoring')):
        return shielded_instance_config_message
    if not args.IsSpecified('shielded_vm_secure_boot') and (not args.IsSpecified('shielded_vm_vtpm')) and (not args.IsSpecified('shielded_vm_integrity_monitoring')):
        return shielded_instance_config_message
    if args.shielded_vm_secure_boot is not None:
        enable_secure_boot = args.shielded_vm_secure_boot
    if args.shielded_vm_vtpm is not None:
        enable_vtpm = args.shielded_vm_vtpm
    if args.shielded_vm_integrity_monitoring is not None:
        enable_integrity_monitoring = args.shielded_vm_integrity_monitoring
    shielded_instance_config_message = instance_utils.CreateShieldedInstanceConfigMessage(messages, enable_secure_boot, enable_vtpm, enable_integrity_monitoring)
    return shielded_instance_config_message