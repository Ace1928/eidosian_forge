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
def _AddWinGroupArguments(parser):
    """Adds Windows setting flags."""
    win_group = parser.add_mutually_exclusive_group(help='Settings for machines running Windows:')
    non_exclusive_group = win_group.add_group(help='Windows patch options')
    non_exclusive_group.add_argument('--windows-classifications', metavar='WINDOWS_CLASSIFICATIONS', type=arg_parsers.ArgList(choices=['critical', 'security', 'definition', 'driver', 'feature-pack', 'service-pack', 'tool', 'update-rollup', 'update']), help='      List of classifications to use to restrict the Windows update. Only\n      patches of the given classifications are applied. If omitted, a default\n      Windows update is performed. For more information on classifications,\n      see: https://support.microsoft.com/en-us/help/824684')
    non_exclusive_group.add_argument('--windows-excludes', metavar='WINDOWS_EXCLUDES', type=arg_parsers.ArgList(), help='Optional list of Knowledge Base (KB) IDs to exclude from the\n      update operation.')
    win_group.add_argument('--windows-exclusive-patches', metavar='WINDOWS_EXCLUSIVE_PATCHES', type=arg_parsers.ArgList(), help='      An exclusive list of Knowledge Base (KB) IDs to be updated. These are the\n      only patches that will be updated.')