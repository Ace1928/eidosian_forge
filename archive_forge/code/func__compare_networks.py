import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def _compare_networks(self, exp, real):
    self.assertDictEqual(_network.Network(**exp).to_dict(computed=False), real.to_dict(computed=False))