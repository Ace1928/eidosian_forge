from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner.resource_args import CloudKmsKeyName
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.credentials import http
from googlecloudsdk.core.util import times
import six
from six.moves import http_client as httplib
from six.moves import urllib
def CopyBackup(source_backup_ref, destination_backup_ref, args, encryption_type=None, kms_key=None):
    """Copy a backup."""
    client = apis.GetClientInstance('spanner', 'v1')
    msgs = apis.GetMessagesModule('spanner', 'v1')
    copy_backup_request = msgs.CopyBackupRequest(backupId=destination_backup_ref.Name(), sourceBackup=source_backup_ref.RelativeName())
    copy_backup_request.expireTime = CheckAndGetExpireTime(args)
    if kms_key:
        copy_backup_request.encryptionConfig = msgs.CopyBackupEncryptionConfig(encryptionType=encryption_type, kmsKeyName=kms_key.kms_key_name, kmsKeyNames=kms_key.kms_key_names)
    elif encryption_type:
        copy_backup_request.encryptionConfig = msgs.CopyBackupEncryptionConfig(encryptionType=encryption_type)
    req = msgs.SpannerProjectsInstancesBackupsCopyRequest(parent=destination_backup_ref.Parent().RelativeName(), copyBackupRequest=copy_backup_request)
    return client.projects_instances_backups.Copy(req)