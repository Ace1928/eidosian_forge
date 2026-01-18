from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.api_lib.alloydb import backup_operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseBackupType(alloydb_messages, backup_type):
    if backup_type:
        return alloydb_messages.Backup.TypeValueValuesEnum.lookup_by_name(backup_type.upper())
    return None