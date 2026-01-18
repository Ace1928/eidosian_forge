import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
class ServersV229Test(ServersV226Test):
    api_version = '2.29'

    def test_evacuate(self):
        s = self.cs.servers.get(1234)
        s.evacuate('fake_target_host')
        self.assert_called('POST', '/servers/1234/action', {'evacuate': {'host': 'fake_target_host'}})
        self.cs.servers.evacuate(s, 'fake_target_host', force=True)
        self.assert_called('POST', '/servers/1234/action', {'evacuate': {'host': 'fake_target_host', 'force': True}})