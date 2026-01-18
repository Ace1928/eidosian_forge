import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareNetworkSubnet(object):
    """Fake a share network subnet"""

    @staticmethod
    def create_one_share_subnet(attrs=None):
        """Create a fake share network subnet

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        share_network_subnet = {'availability_zone': None, 'cidr': '10.0.0.0/24', 'created_at': datetime.datetime.now().isoformat(), 'gateway': '10.0.0.1', 'id': str(uuid.uuid4()), 'ip_version': 4, 'mtu': '1500', 'network_type': 'vlan', 'neutron_net_id': str(uuid.uuid4()), 'neutron_subnet_id': str(uuid.uuid4()), 'segmentation_id': 1010, 'share_network_id': str(uuid.uuid4()), 'share_network_name': str(uuid.uuid4()), 'updated_at': datetime.datetime.now().isoformat(), 'properties': {}}
        share_network_subnet.update(attrs)
        share_network_subnet = osc_fakes.FakeResource(info=copy.deepcopy(share_network_subnet), loaded=True)
        return share_network_subnet

    @staticmethod
    def create_share_network_subnets(attrs=None, count=2):
        """Create multiple fake share network subnets.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share network subnets to be faked
        :return:
            A list of FakeResource objects
        """
        share_network_subnets = []
        for n in range(count):
            share_network_subnets.append(FakeShareNetworkSubnet.create_one_share_subnet(attrs))
        return share_network_subnets