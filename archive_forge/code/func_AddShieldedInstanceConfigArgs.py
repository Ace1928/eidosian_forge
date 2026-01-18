from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddShieldedInstanceConfigArgs(parser, use_default_value=True, for_update=False, for_container=False):
    """Adds flags for Shielded VM configuration.

  Args:
    parser: ArgumentParser, parser to which flags will be added.
    use_default_value: Bool, if True, flag will be given the default value
      False, else None. Update uses None as an indicator that no update needs to
      be done for deletion protection.
    for_update: Bool, if True, flags are intended for an update operation.
    for_container: Bool, if True, flags intended for an instances with container
      operation.
  """
    if use_default_value:
        default_action = 'store_true'
        action_kwargs = {'default': None}
    else:
        default_action = arg_parsers.StoreTrueFalseAction
        action_kwargs = {}
    secure_boot_help = '      The instance boots with secure boot enabled. On Shielded VM instances,\n      Secure Boot is not enabled by default. For information about how to modify\n      Shielded VM options, see\n      https://cloud.google.com/compute/docs/instances/modifying-shielded-vm.\n      '
    if for_update:
        secure_boot_help += '      Changes to this setting with the update command only take effect\n      after stopping and starting the instance.\n      '
    parser.add_argument('--shielded-secure-boot', help=secure_boot_help, dest='shielded_vm_secure_boot', action=default_action, **action_kwargs)
    vtpm_help = '      The instance boots with the TPM (Trusted Platform Module) enabled.\n      A TPM is a hardware module that can be used for different security\n      operations such as remote attestation, encryption, and sealing of keys.\n      On Shielded VM instances, vTPM is enabled by default. For information\n      about how to modify Shielded VM options, see\n      https://cloud.google.com/compute/docs/instances/modifying-shielded-vm.\n      '
    if for_update:
        vtpm_help += '      Changes to this setting with the update command only take effect\n      after stopping and starting the instance.\n      '
    parser.add_argument('--shielded-vtpm', dest='shielded_vm_vtpm', help=vtpm_help, action=default_action, **action_kwargs)
    integrity_monitoring_help_format = '      Enables monitoring and attestation of the boot integrity of the\n      instance. The attestation is performed against the integrity policy\n      baseline. This baseline is initially derived from the implicitly\n      trusted boot image when the instance is created. This baseline can be\n      updated by using\n      `gcloud compute instances {} --shielded-learn-integrity-policy`. On\n      Shielded VM instances, integrity monitoring is enabled by default. For\n      information about how to modify Shielded VM options, see\n      https://cloud.google.com/compute/docs/instances/modifying-shielded-vm.\n      For information about monitoring integrity on Shielded VM instances, see\n      https://cloud.google.com/compute/docs/instances/integrity-monitoring."\n      '
    if for_container:
        update_command = 'update-container'
    else:
        update_command = 'update'
    integrity_monitoring_help = integrity_monitoring_help_format.format(update_command)
    if for_update:
        integrity_monitoring_help += '      Changes to this setting with the update command only take effect\n      after stopping and starting the instance.\n      '
    parser.add_argument('--shielded-integrity-monitoring', help=integrity_monitoring_help, dest='shielded_vm_integrity_monitoring', action=default_action, **action_kwargs)