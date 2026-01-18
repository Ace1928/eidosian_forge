from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _list_share_replicas(self, share=None, search_opts=None):
    """List all share replicas or list replicas belonging to a share.

        :param share: either share object or its UUID.
        :param search_opts: default None
        :rtype: list of :class:`ShareReplica`
        """
    if share:
        share_id = '?share_id=' + base.getid(share)
        url = RESOURCES_PATH + '/detail' + share_id
        return self._list(url, RESOURCES_NAME)
    else:
        return self._list(RESOURCES_PATH + '/detail', RESOURCES_NAME)