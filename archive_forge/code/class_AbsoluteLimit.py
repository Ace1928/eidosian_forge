from openstack import resource
class AbsoluteLimit(resource.Resource):
    max_total_backup_gigabytes = resource.Body('maxTotalBackupGigabytes', type=int)
    max_total_backups = resource.Body('maxTotalBackups', type=int)
    max_total_snapshots = resource.Body('maxTotalSnapshots', type=int)
    max_total_volume_gigabytes = resource.Body('maxTotalVolumeGigabytes', type=int)
    max_total_volumes = resource.Body('maxTotalVolumes', type=int)
    total_backup_gigabytes_used = resource.Body('totalBackupGigabytesUsed', type=int)
    total_backups_used = resource.Body('totalBackupsUsed', type=int)
    total_gigabytes_used = resource.Body('totalGigabytesUsed', type=int)
    total_snapshots_used = resource.Body('totalSnapshotsUsed', type=int)
    total_volumes_used = resource.Body('totalVolumesUsed', type=int)