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
def AddContinuousBackupConfigFlagsForCreateSecondary(parser):
    """Adds Continuous backup configuration flags.

  Args:
    parser: argparse.ArgumentParser: Parser object for command line inputs.
  """
    group = parser.add_group(mutex=False, help='Continuous Backup configuration. If unspecified, continuous backups are copied from the associated primary cluster.')
    group.add_argument('--enable-continuous-backup', action='store_true', default=None, help='Enables Continuous Backups on the cluster.')
    group.add_argument('--continuous-backup-recovery-window-days', metavar='RECOVERY_PERIOD', type=int, help='Recovery window of the log files and backups saved to support Continuous Backups.')
    kms_resource_args.AddKmsKeyResourceArg(group, 'continuous backup', flag_overrides=GetContinuousBackupKmsFlagOverrides(), permission_info="The 'AlloyDB Service Agent's service account must hold permission 'Cloud KMS CryptoKey Encrypter/Decrypter'", name='--continuous-backup-encryption-key')