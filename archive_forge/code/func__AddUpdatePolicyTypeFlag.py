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
def _AddUpdatePolicyTypeFlag(group):
    """Add --update-policy-type flag to the parser."""
    help_text = 'Specifies the type of update process. You can specify either ``proactive`` so that the managed instance group proactively executes actions in order to bring VMs to their target versions or ``opportunistic`` so that no action is proactively executed but the update will be performed as part of other actions.'
    choices = {'opportunistic': 'Do not proactively replace VMs. Create new VMs and delete old ones on resizes of the group and when you target specific VMs to be updated or recreated.', 'proactive': 'Replace VMs proactively.'}
    group.add_argument('--update-policy-type', metavar='UPDATE_TYPE', type=lambda x: x.lower(), choices=choices, help=help_text)