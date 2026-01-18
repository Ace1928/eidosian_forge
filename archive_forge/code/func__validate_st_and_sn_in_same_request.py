from manilaclient import api_versions
from manilaclient import base
def _validate_st_and_sn_in_same_request(self, share_type, share_networks):
    if share_type and share_networks:
        raise ValueError("'share_networks' quota can be set only for project or user, not share type.")