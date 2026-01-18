from manilaclient import api_versions
from manilaclient import base
def _delete_share_group_type(self, share_group_type):
    """Delete a specific share group type.

        :param share_group_type: either instance of ShareGroupType, or text
           with UUID
        """
    share_group_type_id = base.getid(share_group_type)
    url = RESOURCE_PATH % share_group_type_id
    self._delete(url)