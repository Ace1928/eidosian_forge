import argparse
import copy
from random import choice
from random import randint
from unittest import mock
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import address_group as _address_group
from openstack.network.v2 import address_scope as _address_scope
from openstack.network.v2 import agent as network_agent
from openstack.network.v2 import auto_allocated_topology as allocated_topology
from openstack.network.v2 import availability_zone as _availability_zone
from openstack.network.v2 import extension as _extension
from openstack.network.v2 import flavor as _flavor
from openstack.network.v2 import local_ip as _local_ip
from openstack.network.v2 import local_ip_association as _local_ip_association
from openstack.network.v2 import ndp_proxy as _ndp_proxy
from openstack.network.v2 import network as _network
from openstack.network.v2 import network_ip_availability as _ip_availability
from openstack.network.v2 import network_segment_range as _segment_range
from openstack.network.v2 import port as _port
from openstack.network.v2 import rbac_policy as network_rbac
from openstack.network.v2 import segment as _segment
from openstack.network.v2 import service_profile as _flavor_profile
from openstack.network.v2 import trunk as _trunk
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit import utils
class FakeSubnetPool(object):
    """Fake one or more subnet pools."""

    @staticmethod
    def create_one_subnet_pool(attrs=None):
        """Create a fake subnet pool.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object faking the subnet pool
        """
        attrs = attrs or {}
        subnet_pool_attrs = {'id': 'subnet-pool-id-' + uuid.uuid4().hex, 'name': 'subnet-pool-name-' + uuid.uuid4().hex, 'prefixes': ['10.0.0.0/24', '10.1.0.0/24'], 'default_prefixlen': '8', 'address_scope_id': 'address-scope-id-' + uuid.uuid4().hex, 'project_id': 'project-id-' + uuid.uuid4().hex, 'is_default': False, 'shared': False, 'max_prefixlen': '32', 'min_prefixlen': '8', 'default_quota': None, 'ip_version': '4', 'description': 'subnet-pool-description-' + uuid.uuid4().hex, 'tags': [], 'location': 'MUNCHMUNCHMUNCH'}
        subnet_pool_attrs.update(attrs)
        subnet_pool = fakes.FakeResource(info=copy.deepcopy(subnet_pool_attrs), loaded=True)
        subnet_pool.default_prefix_length = subnet_pool_attrs['default_prefixlen']
        subnet_pool.is_shared = subnet_pool_attrs['shared']
        subnet_pool.maximum_prefix_length = subnet_pool_attrs['max_prefixlen']
        subnet_pool.minimum_prefix_length = subnet_pool_attrs['min_prefixlen']
        return subnet_pool

    @staticmethod
    def create_subnet_pools(attrs=None, count=2):
        """Create multiple fake subnet pools.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of subnet pools to fake
        :return:
            A list of FakeResource objects faking the subnet pools
        """
        subnet_pools = []
        for i in range(0, count):
            subnet_pools.append(FakeSubnetPool.create_one_subnet_pool(attrs))
        return subnet_pools

    @staticmethod
    def get_subnet_pools(subnet_pools=None, count=2):
        """Get an iterable Mock object with a list of faked subnet pools.

        If subnet_pools list is provided, then initialize the Mock object
        with the list. Otherwise create one.

        :param List subnet_pools:
            A list of FakeResource objects faking subnet pools
        :param int count:
            The number of subnet pools to fake
        :return:
            An iterable Mock object with side_effect set to a list of faked
            subnet pools
        """
        if subnet_pools is None:
            subnet_pools = FakeSubnetPool.create_subnet_pools(count)
        return mock.Mock(side_effect=subnet_pools)