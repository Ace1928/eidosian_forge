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
class FakeNetworkQosPolicy(object):
    """Fake one or more QoS policies."""

    @staticmethod
    def create_one_qos_policy(attrs=None):
        """Create a fake QoS policy.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object with name, id, etc.
        """
        attrs = attrs or {}
        qos_id = attrs.get('id') or 'qos-policy-id-' + uuid.uuid4().hex
        rule_attrs = {'qos_policy_id': qos_id}
        rules = [FakeNetworkQosRule.create_one_qos_rule(rule_attrs)]
        qos_policy_attrs = {'name': 'qos-policy-name-' + uuid.uuid4().hex, 'id': qos_id, 'is_default': False, 'project_id': 'project-id-' + uuid.uuid4().hex, 'shared': False, 'description': 'qos-policy-description-' + uuid.uuid4().hex, 'rules': rules, 'location': 'MUNCHMUNCHMUNCH'}
        qos_policy_attrs.update(attrs)
        qos_policy = fakes.FakeResource(info=copy.deepcopy(qos_policy_attrs), loaded=True)
        qos_policy.is_shared = qos_policy_attrs['shared']
        return qos_policy

    @staticmethod
    def create_qos_policies(attrs=None, count=2):
        """Create multiple fake QoS policies.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of QoS policies to fake
        :return:
            A list of FakeResource objects faking the QoS policies
        """
        qos_policies = []
        for i in range(0, count):
            qos_policies.append(FakeNetworkQosPolicy.create_one_qos_policy(attrs))
        return qos_policies

    @staticmethod
    def get_qos_policies(qos_policies=None, count=2):
        """Get an iterable MagicMock object with a list of faked QoS policies.

        If qos policies list is provided, then initialize the Mock object
        with the list. Otherwise create one.

        :param List qos_policies:
            A list of FakeResource objects faking qos policies
        :param int count:
            The number of QoS policies to fake
        :return:
            An iterable Mock object with side_effect set to a list of faked
            QoS policies
        """
        if qos_policies is None:
            qos_policies = FakeNetworkQosPolicy.create_qos_policies(count)
        return mock.Mock(side_effect=qos_policies)