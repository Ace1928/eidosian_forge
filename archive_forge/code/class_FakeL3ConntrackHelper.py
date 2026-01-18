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
class FakeL3ConntrackHelper(object):
    """Fake one or more L3 conntrack helper"""

    @staticmethod
    def create_one_l3_conntrack_helper(attrs=None):
        """Create a fake L3 conntrack helper.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object with protocol, port, etc.
        """
        attrs = attrs or {}
        router_id = attrs.get('router_id') or 'router-id-' + uuid.uuid4().hex
        ct_attrs = {'id': uuid.uuid4().hex, 'router_id': router_id, 'helper': 'tftp', 'protocol': 'tcp', 'port': randint(1, 65535), 'location': 'MUNCHMUNCHMUNCH'}
        ct_attrs.update(attrs)
        ct = fakes.FakeResource(info=copy.deepcopy(ct_attrs), loaded=True)
        return ct

    @staticmethod
    def create_l3_conntrack_helpers(attrs=None, count=2):
        """Create multiple fake L3 Conntrack helpers.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of L3 Conntrack helper rule to fake
        :return:
            A list of FakeResource objects faking the Conntrack helpers
        """
        ct_helpers = []
        for i in range(0, count):
            ct_helpers.append(FakeL3ConntrackHelper.create_one_l3_conntrack_helper(attrs))
        return ct_helpers

    @staticmethod
    def get_l3_conntrack_helpers(ct_helpers=None, count=2):
        """Get a list of faked L3 Conntrack helpers.

        If ct_helpers list is provided, then initialize the Mock object
        with the list. Otherwise create one.

        :param List ct_helpers:
            A list of FakeResource objects faking conntrack helpers
        :param int count:
            The number of L3 conntrack helpers to fake
        :return:
            An iterable Mock object with side_effect set to a list of faked
            L3 conntrack helpers
        """
        if ct_helpers is None:
            ct_helpers = FakeL3ConntrackHelper.create_l3_conntrack_helpers(count)
        return mock.Mock(side_effect=ct_helpers)