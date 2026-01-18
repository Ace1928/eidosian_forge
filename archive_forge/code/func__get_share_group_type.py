from manilaclient import api_versions
from manilaclient import base
def _get_share_group_type(self, share_group_type='default'):
    """Get a specific share group type.

        :param share_group_type: either instance of ShareGroupType, or text
           with UUID, or 'default'
        :rtype: :class:`ShareGroupType`
        """
    share_group_type_id = base.getid(share_group_type)
    url = RESOURCE_PATH % share_group_type_id
    return self._get(url, RESOURCE_NAME)