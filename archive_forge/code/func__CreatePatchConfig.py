from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreatePatchConfig(args, messages):
    """Creates a PatchConfig message from input arguments."""
    reboot_config = getattr(messages.PatchConfig.RebootConfigValueValuesEnum, args.reboot_config.upper()) if args.reboot_config else None
    return messages.PatchConfig(rebootConfig=reboot_config, apt=_CreateAptSettings(args, messages), windowsUpdate=_CreateWindowsUpdateSettings(args, messages), yum=_CreateYumSettings(args, messages), zypper=_CreateZypperSettings(args, messages), preStep=_CreatePrePostPatchStepSettings(args, messages, is_pre_patch_step=True), postStep=_CreatePrePostPatchStepSettings(args, messages, is_pre_patch_step=False), migInstancesAllowed=args.mig_instances_allowed)