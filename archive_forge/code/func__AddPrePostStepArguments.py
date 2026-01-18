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
def _AddPrePostStepArguments(parser):
    """Adds pre-/post-patch setting flags."""
    pre_patch_linux_group = parser.add_group(help='Pre-patch step settings for Linux machines:')
    pre_patch_linux_group.add_argument('--pre-patch-linux-executable', help='      A set of commands to run on a Linux machine before an OS patch begins.\n      Commands must be supplied in a file. If the file contains a shell script,\n      include the shebang line.\n\n      The path to the file must be supplied in one of the following formats:\n\n      An absolute path of the file on the local filesystem.\n\n      A URI for a Google Cloud Storage object with a generation number.\n      ')
    pre_patch_linux_group.add_argument('--pre-patch-linux-success-codes', type=arg_parsers.ArgList(element_type=int), metavar='PRE_PATCH_LINUX_SUCCESS_CODES', help='      Additional exit codes that the executable can return to indicate a\n      successful run. The default exit code for success is 0.')
    post_patch_linux_group = parser.add_group(help='Post-patch step settings for Linux machines:')
    post_patch_linux_group.add_argument('--post-patch-linux-executable', help='      A set of commands to run on a Linux machine after an OS patch completes.\n      Commands must be supplied in a file. If the file contains a shell script,\n      include the shebang line.\n\n      The path to the file must be supplied in one of the following formats:\n\n      An absolute path of the file on the local filesystem.\n\n      A URI for a Google Cloud Storage object with a generation number.\n      ')
    post_patch_linux_group.add_argument('--post-patch-linux-success-codes', type=arg_parsers.ArgList(element_type=int), metavar='POST_PATCH_LINUX_SUCCESS_CODES', help='      Additional exit codes that the executable can return to indicate a\n      successful run. The default exit code for success is 0.')
    pre_patch_windows_group = parser.add_group(help='Pre-patch step settings for Windows machines:')
    pre_patch_windows_group.add_argument('--pre-patch-windows-executable', help='      A set of commands to run on a Windows machine before an OS patch begins.\n      Commands must be supplied in a file. If the file contains a PowerShell\n      script, include the .ps1 file extension. The PowerShell script executes\n      with flags `-NonInteractive`, `-NoProfile`, and `-ExecutionPolicy Bypass`.\n\n      The path to the file must be supplied in one of the following formats:\n\n      An absolute path of the file on the local filesystem.\n\n      A URI for a Google Cloud Storage object with a generation number.\n      ')
    pre_patch_windows_group.add_argument('--pre-patch-windows-success-codes', type=arg_parsers.ArgList(element_type=int), metavar='PRE_PATCH_WINDOWS_SUCCESS_CODES', help='      Additional exit codes that the executable can return to indicate a\n      successful run. The default exit code for success is 0.')
    post_patch_windows_group = parser.add_group(help='Post-patch step settings for Windows machines:')
    post_patch_windows_group.add_argument('--post-patch-windows-executable', help='      A set of commands to run on a Windows machine after an OS patch completes.\n      Commands must be supplied in a file. If the file contains a PowerShell\n      script, include the .ps1 file extension. The PowerShell script executes\n      with flags `-NonInteractive`, `-NoProfile`, and `-ExecutionPolicy Bypass`.\n\n      The path to the file must be supplied in one of the following formats:\n\n      An absolute path of the file on the local filesystem.\n\n      A URI for a Google Cloud Storage object with a generation number.\n      ')
    post_patch_windows_group.add_argument('--post-patch-windows-success-codes', type=arg_parsers.ArgList(element_type=int), metavar='POST_PATCH_WINDOWS_SUCCESS_CODES', help='      Additional exit codes that the executable can return to indicate a\n      successful run. The default exit code for success is 0.')