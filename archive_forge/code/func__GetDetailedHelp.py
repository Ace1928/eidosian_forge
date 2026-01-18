from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _GetDetailedHelp(release_track):
    """Formats and returns detailed help for command."""
    detailed_help = {'DESCRIPTION': '      *{command}* creates a patch deployment in a project from a specified file.\n      A patch deployment triggers a patch job to run at specific time(s)\n      according to a schedule, and applies instance filters and other patch\n      configurations to the patch job at run time. Alternatively, to run a patch\n      job on-demand, see *$ gcloud*\n      *{release_track}compute os-config patch-jobs execute*.\n        '.format(command='{command}', release_track=_GetReleaseTrackText(release_track)), 'EXAMPLES': '      To create a patch deployment `patch-deployment-1` in the current project,\n      run:\n\n          $ {command} patch-deployment-1 --file=path_to_config_file\n      '}
    return detailed_help