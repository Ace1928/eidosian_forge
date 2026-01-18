import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
class TestNetworkConnectionSuffix(base.TestCase):

    def test_network_proxy(self):
        self.assertEqual('openstack.network.v2._proxy', self.cloud.network.__class__.__module__)
        self.assert_calls()
        self.assertEqual('https://network.example.com/v2.0', self.cloud.network.get_endpoint())