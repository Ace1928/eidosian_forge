from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _delete_share_group(self, share_group, force=False):
    """Delete a share group.

        :param share_group: either ShareGroup object or text with its UUID
        :param force: True to force the deletion
        """
    share_group_id = base.getid(share_group)
    if force:
        url = RESOURCE_PATH_ACTION % share_group_id
        body = {'force_delete': None}
        self.api.client.post(url, body=body)
    else:
        url = RESOURCE_PATH % share_group_id
        self._delete(url)