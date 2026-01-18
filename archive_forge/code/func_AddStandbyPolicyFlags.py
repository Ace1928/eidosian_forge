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
def AddStandbyPolicyFlags(parser):
    """Add flags required for setting standby policy."""
    standby_policy_mode_choices = {'manual': 'MIG does not automatically resume or start VMs in the standby pool when the group scales out.', 'scale-out-pool': 'MIG automatically resumes or starts VMs in the standby pool when the group scales out, and replenishes the standby pool afterwards.'}
    parser.add_argument('--standby-policy-mode', type=str, choices=standby_policy_mode_choices, help="          Defines how a MIG resumes or starts VMs from a standby pool when the          group scales out. The default mode is ``manual''.\n      ")
    parser.add_argument('--standby-policy-initial-delay', type=int, help='Specifies the number of seconds that the MIG should wait before suspending or stopping a VM. The initial delay gives the initialization script the time to prepare your VM for a quick scale out.')
    parser.add_argument('--suspended-size', type=int, help='Specifies the target size of suspended VMs in the group.')
    parser.add_argument('--stopped-size', type=int, help='Specifies the target size of stopped VMs in the group.')