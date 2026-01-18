import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def _list_shares(self, filters=None):
    filters = filters or dict()
    shares = self.user_client.list_shares(filters=filters)
    self.assertGreater(len(shares), 0)
    if filters:
        for share in shares:
            try:
                share_get = self.user_client.get_share(share['ID'])
            except exceptions.NotFound:
                continue
            if 'migrating' in share_get['status']:
                continue
            for filter_key, expected_value in filters.items():
                if filter_key in ('share_network', 'share-network'):
                    filter_key = 'share_network_id'
                    if share_get[filter_key] != expected_value:
                        self.assertNotIn(share_get['id'], self.shares_created)
                        continue
                if expected_value != 'deleting' and share_get[filter_key] == 'deleting':
                    continue
                self.assertEqual(expected_value, share_get[filter_key])