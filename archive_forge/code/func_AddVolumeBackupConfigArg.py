from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeBackupConfigArg(parser):
    """Adds the --backup-config arg to the arg parser."""
    backup_config_arg_spec = {'backup-policies': arg_parsers.ArgList(min_length=1, element_type=str, custom_delim_char='#'), 'backup-vault': str, 'enable-scheduled-backups': arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey)}
    backup_config_help = 'Backup Config contains backup related config on a volume.\n\n    Backup Config will have the following format\n    `--backup-config=backup-policies=BACKUP_POLICIES,\n    backup-vault=BACKUP_VAULT_NAME,\n    enable-scheduled-backups=ENABLE_SCHEDULED_BACKUPS\n\nbackup-policies is a pound-separated (#) list of backup policy names, backup-vault can include\na single backup-vault resource name, and enable-scheduled-backups is a Boolean value indicating\nwhether or not scheduled backups are enabled on the volume.\n  '
    parser.add_argument('--backup-config', type=arg_parsers.ArgDict(spec=backup_config_arg_spec), help=backup_config_help)