from manilaclient import api_versions
from manilaclient import base
def _check_user_id_and_share_type_args(self, user_id, share_type):
    if user_id and share_type:
        raise ValueError("'user_id' and 'share_type' values are mutually exclusive. one or both should be unset.")