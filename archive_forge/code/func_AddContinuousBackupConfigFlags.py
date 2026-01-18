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
def AddContinuousBackupConfigFlags(parser, release_track, update=False):
    """Adds Continuous backup configuration flags.

  Args:
    parser: argparse.ArgumentParser: Parser object for command line inputs.
    release_track: The command version being used - GA/BETA/ALPHA.
    update: Whether database flags were provided as part of an update.
  """
    continuous_backup_help = 'Continuous Backup configuration.'
    if not update:
        continuous_backup_help += ' If unspecified, continuous backups are enabled.'
    group = parser.add_group(mutex=False, help=continuous_backup_help)
    group.add_argument('--enable-continuous-backup', action='store_true', default=None, help='Enables Continuous Backups on the cluster.')
    group.add_argument('--continuous-backup-recovery-window-days', metavar='RECOVERY_PERIOD', type=int, help='Recovery window of the log files and backups saved to support Continuous Backups.')
    if release_track == base.ReleaseTrack.ALPHA or release_track == base.ReleaseTrack.BETA:
        group.add_argument('--continuous-backup-enforced-retention', action='store_true', default=None, required=False, hidden=True, help='If set, enforces the retention period for continuous backups. Backups created by this configuration cannot be deleted before they are out of retention.')
    cmek_group = group
    if update:
        cmek_group = group.add_group(mutex=True, help='Encryption configuration for Continuous Backups.')
        cmek_group.add_argument('--clear-continuous-backup-encryption-key', action='store_true', help='Clears the encryption configuration for Continuous Backups. Google default encryption will be used for future Continuous Backups.')
    kms_resource_args.AddKmsKeyResourceArg(cmek_group, 'continuous backup', flag_overrides=GetContinuousBackupKmsFlagOverrides(), permission_info="The 'AlloyDB Service Agent's service account must hold permission 'Cloud KMS CryptoKey Encrypter/Decrypter'", name='--continuous-backup-encryption-key')