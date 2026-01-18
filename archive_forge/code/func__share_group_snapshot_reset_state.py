from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _share_group_snapshot_reset_state(self, share_group_snapshot, state):
    """Update the specified share group snapshot.

        :param share_group_snapshot: either ShareGroupSnapshot object or text
            with its UUID
        :param state: The new state for the share group snapshot
        """
    share_group_snapshot_id = base.getid(share_group_snapshot)
    url = RESOURCE_PATH_ACTION % share_group_snapshot_id
    body = {'reset_status': {'status': state}}
    self.api.client.post(url, body=body)