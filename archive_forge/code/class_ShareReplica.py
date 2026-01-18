from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
class ShareReplica(base.Resource):
    """A replica is 'mirror' instance of a share at some point in time."""

    def __repr__(self):
        return '<Share Replica: %s>' % self.id

    def resync(self):
        """Re-sync this replica."""
        self.manager.resync(self)

    def promote(self):
        """Promote this replica to be the 'active' replica."""
        self.manager.promote(self)

    def reset_state(self, state):
        """Update replica's 'status' attr with the provided state."""
        self.manager.reset_state(self, state)

    def reset_replica_state(self, replica_state):
        """Update replica's 'replica_state' attr with the provided state."""
        self.manager.reset_replica_state(self, replica_state)