from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _list_share_group_snapshots(self, detailed=True, search_opts=None, sort_key=None, sort_dir=None):
    """Get a list of all share group snapshots.

        :param detailed: Whether to return detailed snapshot info or not.
        :param search_opts: dict with search options to filter out snapshots.
            available keys are below (('name1', 'name2', ...), 'type'):
            - ('all_tenants', int)
            - ('offset', int)
            - ('limit', int)
            - ('name', text)
            - ('status', text)
            - ('share_group_id', text)
        :param sort_key: Key to be sorted (i.e. 'created_at' or 'status').
        :param sort_dir: Sort direction, should be 'desc' or 'asc'.
        :rtype: list of :class:`ShareGroupSnapshot`
        """
    search_opts = search_opts or {}
    if sort_key is not None:
        if sort_key in constants.SHARE_GROUP_SNAPSHOT_SORT_KEY_VALUES:
            search_opts['sort_key'] = sort_key
        else:
            msg = 'sort_key must be one of the following: %s.'
            msg_args = ', '.join(constants.SHARE_GROUP_SNAPSHOT_SORT_KEY_VALUES)
            raise ValueError(msg % msg_args)
    if sort_dir is not None:
        if sort_dir in constants.SORT_DIR_VALUES:
            search_opts['sort_dir'] = sort_dir
        else:
            raise ValueError('sort_dir must be one of the following: %s.' % ', '.join(constants.SORT_DIR_VALUES))
    query_string = self._build_query_string(search_opts)
    if detailed:
        url = RESOURCES_PATH + '/detail' + query_string
    else:
        url = RESOURCES_PATH + query_string
    return self._list(url, RESOURCES_NAME)