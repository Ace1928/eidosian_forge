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
class FakeSecurityGroupRule(object):
    """Fake one or more security group rules."""

    @staticmethod
    def create_one_security_group_rule(attrs=None):
        """Create a fake security group rule.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, etc.
        """
        attrs = attrs or {}
        security_group_rule_attrs = {'description': 'security-group-rule-description-' + uuid.uuid4().hex, 'direction': 'ingress', 'ether_type': 'IPv4', 'id': 'security-group-rule-id-' + uuid.uuid4().hex, 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_address_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'security_group_id': 'security-group-id-' + uuid.uuid4().hex, 'project_id': 'project-id-' + uuid.uuid4().hex, 'location': 'MUNCHMUNCHMUNCH'}
        security_group_rule_attrs.update(attrs)
        security_group_rule = fakes.FakeResource(info=copy.deepcopy(security_group_rule_attrs), loaded=True)
        return security_group_rule

    @staticmethod
    def create_security_group_rules(attrs=None, count=2):
        """Create multiple fake security group rules.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of security group rules to fake
        :return:
            A list of FakeResource objects faking the security group rules
        """
        security_group_rules = []
        for i in range(0, count):
            security_group_rules.append(FakeSecurityGroupRule.create_one_security_group_rule(attrs))
        return security_group_rules

    @staticmethod
    def get_security_group_rules(security_group_rules=None, count=2):
        """Get an iterable Mock with a list of faked security group rules.

        If security group rules list is provided, then initialize the Mock
        object with the list. Otherwise create one.

        :param List security_group_rules:
            A list of FakeResource objects faking security group rules
        :param int count:
            The number of security group rules to fake
        :return:
            An iterable Mock object with side_effect set to a list of faked
            security group rules
        """
        if security_group_rules is None:
            security_group_rules = FakeSecurityGroupRule.create_security_group_rules(count)
        return mock.Mock(side_effect=security_group_rules)