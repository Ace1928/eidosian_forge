from unittest import mock
from unittest.mock import call
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import (
from openstack.test import fakes as sdk_fakes
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import default_security_group_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def _setup_default_security_group_rule(self, attrs=None):
    default_security_group_rule_attrs = {'description': 'default-security-group-rule-description-' + uuid.uuid4().hex, 'direction': 'ingress', 'ether_type': 'IPv4', 'id': 'default-security-group-rule-id-' + uuid.uuid4().hex, 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_address_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'location': 'MUNCHMUNCHMUNCH', 'used_in_default_sg': False, 'used_in_non_default_sg': True}
    attrs = attrs or {}
    default_security_group_rule_attrs.update(attrs)
    self._default_sg_rule = sdk_fakes.generate_fake_resource(_default_security_group_rule.DefaultSecurityGroupRule, **default_security_group_rule_attrs)
    self.sdk_client.create_default_security_group_rule.return_value = self._default_sg_rule
    self.expected_data = (self._default_sg_rule.description, self._default_sg_rule.direction, self._default_sg_rule.ether_type, self._default_sg_rule.id, self._default_sg_rule.port_range_max, self._default_sg_rule.port_range_min, self._default_sg_rule.protocol, self._default_sg_rule.remote_address_group_id, self._default_sg_rule.remote_group_id, self._default_sg_rule.remote_ip_prefix, self._default_sg_rule.used_in_default_sg, self._default_sg_rule.used_in_non_default_sg)