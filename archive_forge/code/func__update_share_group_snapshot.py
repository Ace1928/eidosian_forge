from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _update_share_group_snapshot(self, share_group_snapshot, **kwargs):
    """Updates a share group snapshot.

        :param share_group_snapshot: either ShareGroupSnapshot object or text
            with its UUID
        :rtype: :class:`ShareGroupSnapshot`
        """
    share_group_snapshot_id = base.getid(share_group_snapshot)
    url = RESOURCE_PATH % share_group_snapshot_id
    if not kwargs:
        return self._get(url, RESOURCE_NAME)
    else:
        body = {RESOURCE_NAME: kwargs}
        return self._update(url, body, RESOURCE_NAME)