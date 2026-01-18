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
def AddMigUpdatePolicyFlags(parser, support_min_ready_flag=False):
    """Add flags required for setting update policy attributes."""
    group = parser.add_group(required=False, mutex=False, help='Parameters for setting update policy for this managed instance group.')
    _AddUpdatePolicyTypeFlag(group)
    _AddUpdatePolicyMaxUnavailableFlag(group)
    _AddUpdatePolicyMaxSurgeFlag(group)
    _AddUpdatePolicyMinimalActionFlag(group)
    _AddUpdatePolicyMostDisruptiveActionFlag(group)
    _AddUpdatePolicyReplacementMethodFlag(group)
    if support_min_ready_flag:
        _AddUpdatePolicyMinReadyFlag(group)