import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _list_share_networks_with_filters(self, filters):
    assert_subnet_fields = utils.share_network_subnets_are_supported()
    share_subnet_fields = ['neutron_subnet_id', 'neutron_net_id'] if assert_subnet_fields else []
    share_network_filters = [(k, v) for k, v in filters.items() if k not in share_subnet_fields]
    share_network_subnet_filters = [(k, v) for k, v in filters.items() if k in share_subnet_fields]
    share_networks = self.admin_client.list_share_networks(filters=filters)
    self.assertGreater(len(share_networks), 0)
    self.assertTrue(any((self.sn['id'] == sn['id'] for sn in share_networks)))
    for sn in share_networks:
        try:
            share_network = self.admin_client.get_share_network(sn['id'])
            default_subnet = utils.get_default_subnet(self.user_client, sn['id']) if assert_subnet_fields else None
        except tempest_lib_exc.NotFound:
            continue
        for k, v in share_network_filters:
            self.assertIn(k, share_network)
            self.assertEqual(v, share_network[k])
        for k, v in share_network_subnet_filters:
            self.assertIn(k, default_subnet)
            self.assertEqual(v, default_subnet[k])