from manilaclient import api_versions
from manilaclient import base
class ShareSnapshotExportLocation(base.Resource):
    """Represent an export location snapshot of a snapshot."""

    def __repr__(self):
        return '<ShareSnapshotExportLocation: %s>' % self.id

    def __getitem__(self, key):
        return self._info[key]