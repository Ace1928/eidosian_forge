from manilaclient import api_versions
from manilaclient import base
def _list_share_group_types(self, show_all=True, search_opts=None):
    """Get a list of all share group types.

        :rtype: list of :class:`ShareGroupType`.
        """
    search_opts = search_opts or {}
    if show_all:
        search_opts['is_public'] = 'all'
    query_string = self._build_query_string(search_opts)
    url = RESOURCES_PATH + query_string
    return self._list(url, RESOURCES_NAME)