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
class TestNewService(base.TestCase):

    def test_add_service_v1(self):
        svc = self.os_fixture.v3_token.add_service('fake')
        svc.add_endpoint(interface='public', region='RegionOne', url='https://fake.example.com/v1/{0}'.format(fakes.PROJECT_ID))
        self.use_keystone_v3()
        conn = self.cloud
        service = fake_service.FakeService('fake')
        conn.add_service(service)
        self.assertEqual(0, len(self.adapter.request_history))
        self.register_uris([dict(method='GET', uri='https://fake.example.com', status_code=404), dict(method='GET', uri='https://fake.example.com/v1/', status_code=404), dict(method='GET', uri=self.get_mock_url('fake'), status_code=404)])
        self.assertEqual('openstack.tests.unit.fake.v1._proxy', conn.fake.__class__.__module__)
        self.assertTrue(conn.fake.dummy())

    def test_add_service_v2(self):
        svc = self.os_fixture.v3_token.add_service('fake')
        svc.add_endpoint(interface='public', region='RegionOne', url='https://fake.example.com/v2/{0}'.format(fakes.PROJECT_ID))
        self.use_keystone_v3()
        conn = self.cloud
        self.register_uris([dict(method='GET', uri='https://fake.example.com', status_code=404), dict(method='GET', uri='https://fake.example.com/v2/', status_code=404), dict(method='GET', uri=self.get_mock_url('fake'), status_code=404)])
        service = fake_service.FakeService('fake')
        conn.add_service(service)
        self.assertEqual('openstack.tests.unit.fake.v2._proxy', conn.fake.__class__.__module__)
        self.assertFalse(conn.fake.dummy())

    def test_replace_system_service(self):
        svc = self.os_fixture.v3_token.add_service('fake')
        svc.add_endpoint(interface='public', region='RegionOne', url='https://fake.example.com/v2/{0}'.format(fakes.PROJECT_ID))
        self.use_keystone_v3()
        conn = self.cloud
        delattr(conn, 'dns')
        self.register_uris([dict(method='GET', uri='https://fake.example.com', status_code=404), dict(method='GET', uri='https://fake.example.com/v2/', status_code=404), dict(method='GET', uri=self.get_mock_url('fake'), status_code=404)])
        service = fake_service.FakeService('fake', aliases=['dns'])
        conn.add_service(service)
        self.assertFalse(conn.dns.dummy())