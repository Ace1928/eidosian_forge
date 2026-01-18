from manilaclient import api_versions
from manilaclient import base
def _list_share_group_type_access(self, share_group_type, search_opts=None):
    if share_group_type.is_public:
        return None
    share_group_type_id = base.getid(share_group_type)
    url = RESOURCE_PATH % share_group_type_id
    return self._list(url, RESOURCE_NAME)