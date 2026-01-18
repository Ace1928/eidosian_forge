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
def ValidateContinuousBackupFlags(args, update=False):
    """Validate the arguments for continuous backup, ensure the correct set of flags are passed."""
    if args.enable_continuous_backup is False and (args.continuous_backup_recovery_window_days or args.continuous_backup_encryption_key or (update and args.clear_continuous_backup_encryption_key)):
        raise exceptions.ConflictingArgumentsException('--no-enable-continuous-backup', '--continuous-backup-recovery-window-days', '--continuous-backup-encryption-key', '--clear-continuous-backup-encryption-key')