from tempest.lib import exceptions as tempest_lib_exc
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _test_create_list_access_rule_for_snapshot(self, snapshot_id):
    access = []
    access_type = self.access_types[0]
    for i in range(5):
        access_ = self.user_client.snapshot_access_allow(snapshot_id, access_type, self.access_to[access_type][i])
        access.append(access_)
    return access