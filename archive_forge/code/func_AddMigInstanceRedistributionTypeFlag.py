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
def AddMigInstanceRedistributionTypeFlag(parser):
    """Add --instance-redistribution-type flag to the parser."""
    help_text = "      Specifies the type of the instance redistribution policy. An instance\n      redistribution type lets you enable or disable automatic instance\n      redistribution across zones to meet the group's target distribution shape.\n\n      An instance redistribution type can be specified only for a non-autoscaled\n      regional managed instance group. By default it is set to ``proactive''.\n      "
    choices = {'none': 'The managed instance group does not redistribute instances across zones.', 'proactive': 'The managed instance group proactively redistributes instances to meet its target distribution.'}
    parser.add_argument('--instance-redistribution-type', metavar='TYPE', type=lambda x: x.lower(), choices=choices, help=help_text)