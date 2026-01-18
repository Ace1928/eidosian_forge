from openstack import exceptions
from openstack.network.v2 import qos_rule_type
from openstack.tests.unit import base
def _compare_rule_types(self, exp, real):
    self.assertDictEqual(qos_rule_type.QoSRuleType(**exp).to_dict(computed=False), real.to_dict(computed=False))