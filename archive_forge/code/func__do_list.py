from manilaclient import api_versions
from manilaclient import base
def _do_list(self, share_type, action_name='share_type_access'):
    if share_type.is_public:
        return None
    return self._list('/types/%(st_id)s/%(action_name)s' % {'st_id': base.getid(share_type), 'action_name': action_name}, 'share_type_access')