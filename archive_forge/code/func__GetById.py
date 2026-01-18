from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
def _GetById(self, backup_id, instance_name, project_level):
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    if project_level:
        request = sql_messages.SqlBackupsGetBackupRequest(name=backup_id)
        return sql_client.backups.GetBackup(request)
    instance_ref = client.resource_parser.Parse(instance_name, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    request = sql_messages.SqlBackupRunsGetRequest(project=instance_ref.project, instance=instance_ref.instance, id=int(backup_id))
    return sql_client.backupRuns.Get(request)