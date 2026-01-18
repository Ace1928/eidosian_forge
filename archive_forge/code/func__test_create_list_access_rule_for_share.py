import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
def _test_create_list_access_rule_for_share(self, microversion, metadata=None):
    access_type = self.access_types[0]
    access = self.user_client.access_allow(self.share['id'], access_type, self.access_to[access_type].pop(), self.access_level, metadata=metadata, microversion=microversion)
    return access