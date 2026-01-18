from manilaclient import api_versions
from manilaclient import base
class ShareSnapshotInstanceExportLocation(base.Resource):
    """Represent an export location from a snapshot instance."""

    def __repr__(self):
        return '<ShareSnapshotInstanceExportLocation: %s>' % self.id

    def __getitem__(self, key):
        return self._info[key]