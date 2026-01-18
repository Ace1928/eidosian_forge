import copy
from openstack import exceptions
from openstack.network.v2 import qos_minimum_bandwidth_rule
from openstack.tests.unit import base
def _compare_rules(self, exp, real):
    self.assertDictEqual(qos_minimum_bandwidth_rule.QoSMinimumBandwidthRule(**exp).to_dict(computed=False), real.to_dict(computed=False))