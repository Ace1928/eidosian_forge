import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
class TestNetworkAndComputeCommand(TestNetworkAndCompute):

    def setUp(self):
        super(TestNetworkAndComputeCommand, self).setUp()
        self.cmd = FakeNetworkAndComputeCommand(self.app, self.namespace)