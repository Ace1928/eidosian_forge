from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetShieldedInstanceConfigFromArgs(args, messages):
    """Creates the Shielded Instance Config message for the create/update request.

  Args:
    args: Argparse object from Command.Run
    messages: Module containing messages definition for the specified API.

  Returns:
    Shielded Instance Config of the Instance message.
  """
    if not (args.IsSpecified('shielded_secure_boot') or args.IsSpecified('shielded_vtpm') or args.IsSpecified('shielded_integrity_monitoring')):
        return None
    true_values = ['1', 'true', 'on', 'yes', 'y']
    if args.IsSpecified('shielded_secure_boot'):
        shielded_secure_boot = args.shielded_secure_boot.lower() in true_values
    else:
        shielded_secure_boot = False
    if args.IsSpecified('shielded_vtpm'):
        shielded_vtpm = args.shielded_vtpm.lower() in true_values
    else:
        shielded_vtpm = True
    if args.IsSpecified('shielded_integrity_monitoring'):
        shielded_integrity_monitoring = args.shielded_integrity_monitoring.lower() in true_values
    else:
        shielded_integrity_monitoring = True
    shielded_instance_config_message = messages.ShieldedInstanceConfig
    return shielded_instance_config_message(enableIntegrityMonitoring=shielded_integrity_monitoring, enableSecureBoot=shielded_secure_boot, enableVtpm=shielded_vtpm)