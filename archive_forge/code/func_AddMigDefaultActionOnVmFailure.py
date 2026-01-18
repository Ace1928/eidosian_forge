from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMigDefaultActionOnVmFailure(parser):
    """Add default action on VM failure to the parser."""
    help_text = "      Specifies the action that a MIG performs on a failed or an unhealthy VM.\n      A VM is marked as unhealthy when the application running on that VM\n      fails a health check.\n      By default, the value of the flag is set to ``repair''."
    choices = {'repair': 'MIG automatically repairs a failed or an unhealthy VM.', 'do-nothing': 'MIG does not repair a failed or an unhealthy VM.'}
    parser.add_argument('--default-action-on-vm-failure', metavar='ACTION_ON_VM_FAILURE', type=arg_utils.EnumNameToChoice, choices=choices, help=help_text)