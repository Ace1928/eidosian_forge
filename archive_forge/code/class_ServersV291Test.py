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
class ServersV291Test(ServersV290Test):
    api_version = '2.91'

    def test_unshelve_with_host(self):
        s = self.cs.servers.get(1234)
        ret = s.unshelve(host='server1')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'host': 'server1'}})
        ret = self.cs.servers.unshelve(s, host='server1')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'host': 'server1'}})

    def test_unshelve_server_with_az_and_host(self):
        s = self.cs.servers.get(1234)
        ret = s.unshelve(host='server1', availability_zone='foo-az')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'host': 'server1', 'availability_zone': 'foo-az'}})
        ret = self.cs.servers.unshelve(s, host='server1', availability_zone='foo-az')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'host': 'server1', 'availability_zone': 'foo-az'}})

    def test_unshelve_unpin_az(self):
        s = self.cs.servers.get(1234)
        ret = s.unshelve(availability_zone=None)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'availability_zone': None}})
        ret = self.cs.servers.unshelve(s, availability_zone=None)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'availability_zone': None}})

    def test_unshelve_server_with_host_and_unpin(self):
        s = self.cs.servers.get(1234)
        ret = s.unshelve(availability_zone=None, host='server1')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'host': 'server1', 'availability_zone': None}})
        ret = self.cs.servers.unshelve(s, availability_zone=None, host='server1')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'host': 'server1', 'availability_zone': None}})