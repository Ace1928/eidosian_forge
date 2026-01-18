from cinderclient import base
class VolumeBackupsRestore(base.Resource):
    """A Volume Backups Restore represents a restore operation."""

    def __repr__(self):
        return '<VolumeBackupsRestore: %s>' % self.volume_id