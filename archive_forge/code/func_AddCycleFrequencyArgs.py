from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddCycleFrequencyArgs(parser, flag_suffix, start_time_help, cadence_help, supports_hourly=False, has_restricted_start_times=False, supports_weekly=False, required=True):
    """Add Cycle Frequency args for Resource Policies."""
    freq_group = parser.add_argument_group('Cycle Frequency Group.', required=required, mutex=True)
    if has_restricted_start_times:
        start_time_help += '        Valid choices are 00:00, 04:00, 08:00, 12:00,\n        16:00 and 20:00 UTC. For example, `--start-time="08:00"`.'
    freq_flags_group = freq_group.add_group('Using command flags:' if supports_weekly else '')
    freq_flags_group.add_argument('--start-time', required=True, type=arg_parsers.Datetime.ParseUtcTime, help=start_time_help)
    cadence_group = freq_flags_group.add_group(mutex=True, required=True)
    cadence_group.add_argument('--daily-{}'.format(flag_suffix), dest='daily_cycle', action='store_true', help='{} starts daily at START_TIME.'.format(cadence_help))
    if supports_hourly:
        cadence_group.add_argument('--hourly-{}'.format(flag_suffix), metavar='HOURS', dest='hourly_cycle', type=arg_parsers.BoundedInt(lower_bound=1), help='{} occurs every n hours starting at START_TIME.'.format(cadence_help))
    if supports_weekly:
        base.ChoiceArgument('--weekly-{}'.format(flag_suffix), dest='weekly_cycle', choices=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], help_str='{} occurs weekly on WEEKLY_{} at START_TIME.'.format(cadence_help, flag_suffix.upper())).AddToParser(cadence_group)
        freq_file_group = freq_group.add_group('Using a file:')
        freq_file_group.add_argument('--weekly-{}-from-file'.format(flag_suffix), dest='weekly_cycle_from_file', type=arg_parsers.FileContents(), help='        A JSON/YAML file which specifies a weekly schedule. The file should\n        contain the following fields:\n\n        day: Day of the week with the same choices as `--weekly-{}`.\n        startTime: Start time of the snapshot schedule with\n        the same format as --start-time.\n\n        For more information about using a file,\n        see https://cloud.google.com/compute/docs/disks/scheduled-snapshots#create_snapshot_schedule\n        '.format(flag_suffix))