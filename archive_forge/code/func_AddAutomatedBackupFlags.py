from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAutomatedBackupFlags(parser, alloydb_messages, release_track, update=False):
    """Adds automated backup flags.

  Args:
    parser: argparse.ArgumentParser: Parser object for command line inputs.
    alloydb_messages: Message module.
    release_track: The command version being used - GA/BETA/ALPHA.
    update: If True, adds update specific flags.
  """
    automated_backup_help = 'Automated backup policy.'
    if not update:
        automated_backup_help += ' If unspecified, automated backups are disabled.'
    group = parser.add_group(mutex=True, help=automated_backup_help)
    policy_group = group.add_group(help='Enable automated backup policy.')
    policy_group.add_argument('--automated-backup-days-of-week', metavar='DAYS_OF_WEEK', required=not update, type=_GetDayOfWeekArgList(alloydb_messages), help='Comma-separated list of days of the week to perform a backup. At least one day of the week must be provided. (e.g., --automated-backup-days-of-week=MONDAY,WEDNESDAY,SUNDAY)')
    policy_group.add_argument('--automated-backup-start-times', metavar='START_TIMES', required=not update, type=_GetTimeOfDayArgList(alloydb_messages), help='Comma-separated list of times during the day to start a backup. At least one start time must be provided. The start times are assumed to be in UTC and required to be an exact hour in the format HH:00. (e.g., `--automated-backup-start-times=01:00,13:00`)')
    retention_group = policy_group.add_group(mutex=True, help='Retention policy. If no retention policy is provided, all automated backups will be retained.')
    retention_group.add_argument('--automated-backup-retention-period', metavar='RETENTION_PERIOD', type=arg_parsers.Duration(parsed_unit='s'), help='Retention period of the backup relative to creation time.  See `$ gcloud topic datetimes` for information on duration formats.')
    retention_group.add_argument('--automated-backup-retention-count', metavar='RETENTION_COUNT', type=int, help='Number of most recent successful backups retained.')
    policy_group.add_argument('--automated-backup-window', metavar='TIMEOUT_PERIOD', type=arg_parsers.Duration(lower_bound='5m', parsed_unit='s'), help='The length of the time window beginning at start time during which a backup can be taken. If a backup does not succeed within this time window, it will be canceled and considered failed. The backup window must be at least 5 minutes long. There is no upper bound on the window. If not set, it will default to 1 hour.')
    if release_track == base.ReleaseTrack.ALPHA or release_track == base.ReleaseTrack.BETA:
        policy_group.add_argument('--automated-backup-enforced-retention', action='store_true', default=None, required=False, hidden=True, help='If set, enforces the retention period for automated backups. Backups created by this policy cannot be deleted before they are out of retention.')
    kms_resource_args.AddKmsKeyResourceArg(policy_group, 'automated backups', flag_overrides=GetAutomatedBackupKmsFlagOverrides(), permission_info="The 'AlloyDB Service Agent' service account must hold permission 'Cloud KMS CryptoKey Encrypter/Decrypter'", name='--automated-backup-encryption-key')
    if update:
        group.add_argument('--clear-automated-backup', action='store_true', help='Clears the automated backup policy on the cluster. The default automated backup policy will be used.')
    group.add_argument('--disable-automated-backup', action='store_true', help='Disables automated backups on the cluster.')