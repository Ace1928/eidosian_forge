from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _share_group_reset_state(self, share_group, state):
    """Update the specified share group with the provided state.

        :param share_group: either ShareGroup object or text with its UUID
        :param state: The new state for the share group
        """
    share_group_id = base.getid(share_group)
    url = RESOURCE_PATH_ACTION % share_group_id
    body = {'reset_status': {'status': state}}
    self.api.client.post(url, body=body)