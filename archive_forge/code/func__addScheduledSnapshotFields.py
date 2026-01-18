from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import environment_patch_util as patch_util
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
def _addScheduledSnapshotFields(self, params, args, is_composer_v1):
    if (args.disable_scheduled_snapshot_creation or args.enable_scheduled_snapshot_creation) and is_composer_v1:
        raise command_util.InvalidUserInputError('Scheduled Snapshots flags introduced in Composer 2.X cannot be used when creating Composer 1 environments.')
    if args.disable_scheduled_snapshot_creation:
        params['enable_scheduled_snapshot_creation'] = False
    if args.enable_scheduled_snapshot_creation:
        params['enable_scheduled_snapshot_creation'] = True
        params['snapshot_location'] = args.snapshot_location
        params['snapshot_schedule_timezone'] = args.snapshot_schedule_timezone
        params['snapshot_creation_schedule'] = args.snapshot_creation_schedule