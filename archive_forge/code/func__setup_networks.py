import pprint
import sys
from testtools import content
from openstack.cloud import meta
from openstack import exceptions
from openstack import proxy
from openstack.tests.functional import base
from openstack import utils
def _setup_networks(self):
    if self.user_cloud.has_service('network'):
        self.test_net = self.user_cloud.create_network(name=self.new_item_name + '_net')
        self.test_subnet = self.user_cloud.create_subnet(subnet_name=self.new_item_name + '_subnet', network_name_or_id=self.test_net['id'], cidr='10.24.4.0/24', enable_dhcp=True)
        self.test_router = self.user_cloud.create_router(name=self.new_item_name + '_router')
        ext_nets = self.user_cloud.search_networks(filters={'router:external': True})
        self.user_cloud.update_router(name_or_id=self.test_router['id'], ext_gateway_net_id=ext_nets[0]['id'])
        self.user_cloud.add_router_interface(self.test_router, subnet_id=self.test_subnet['id'])
        self.nic = {'net-id': self.test_net['id']}
        self.addDetail('networks-neutron', content.text_content(pprint.pformat(self.user_cloud.list_networks())))
    else:
        data = proxy._json_response(self.user_cloud._conn.compute.get('/os-tenant-networks'))
        nets = meta.get_and_munchify('networks', data)
        self.addDetail('networks-nova', content.text_content(pprint.pformat(nets)))
        self.nic = {'net-id': nets[0].id}