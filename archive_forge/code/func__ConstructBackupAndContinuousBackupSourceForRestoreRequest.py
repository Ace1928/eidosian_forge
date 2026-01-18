from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructBackupAndContinuousBackupSourceForRestoreRequest(alloydb_messages, resource_parser, args):
    """Returns the backup and continuous backup source for restore request."""
    backup_source, continuous_backup_source = (None, None)
    if args.backup:
        backup_ref = resource_parser.Parse(collection='alloydb.projects.locations.backups', line=args.backup, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'locationsId': args.region})
        backup_source = alloydb_messages.BackupSource(backupName=backup_ref.RelativeName())
    else:
        cluster_ref = resource_parser.Parse(collection='alloydb.projects.locations.clusters', line=args.source_cluster, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'locationsId': args.region})
        continuous_backup_source = alloydb_messages.ContinuousBackupSource(cluster=cluster_ref.RelativeName(), pointInTime=args.point_in_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
    return (backup_source, continuous_backup_source)