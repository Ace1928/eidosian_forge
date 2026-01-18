from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def AddNamedPortsArgs(parser):
    """Adds flags for handling named ports."""
    parser.add_argument('--named-ports', required=True, type=arg_parsers.ArgList(), metavar='NAME:PORT', help='          The comma-separated list of key:value pairs representing\n          the service name and the port that it is running on.\n\n          To clear the list of named ports pass empty list as flag value.\n          For example:\n\n            $ {command} example-instance-group --named-ports ""\n          ')