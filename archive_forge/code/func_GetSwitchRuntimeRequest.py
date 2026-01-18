from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetSwitchRuntimeRequest(args, messages):
    """Create and return switch runtime request."""
    machine_type = 'n1-standard-4'
    if args.IsSpecified('machine_type'):
        machine_type = args.machine_type
    runtime_accelerator_config = messages.RuntimeAcceleratorConfig()
    if args.IsSpecified('accelerator_core_count'):
        runtime_accelerator_config.coreCount = args.accelerator_core_count
    if args.IsSpecified('accelerator_type'):
        type_enum = arg_utils.ChoiceEnumMapper(arg_name='accelerator-type', message_enum=runtime_accelerator_config.TypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.accelerator_type))
        runtime_accelerator_config.type = type_enum
    return messages.SwitchRuntimeRequest(machineType=machine_type, acceleratorConfig=runtime_accelerator_config)