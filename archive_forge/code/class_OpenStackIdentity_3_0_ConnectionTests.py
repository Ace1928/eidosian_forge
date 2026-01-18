import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
class OpenStackIdentity_3_0_ConnectionTests(unittest.TestCase):

    def setUp(self):
        mock_cls = OpenStackIdentity_3_0_MockHttp
        mock_cls.type = None
        OpenStackIdentity_3_0_Connection.conn_class = mock_cls
        self.auth_instance = OpenStackIdentity_3_0_Connection(auth_url='http://none', user_id='test', key='test', tenant_name='test', proxy_url='http://proxy:8080', timeout=10)
        self.auth_instance.auth_token = 'mock'
        self.assertEqual(self.auth_instance.proxy_url, 'http://proxy:8080')

    def test_token_scope_argument(self):
        expected_msg = 'Invalid value for "token_scope" argument: foo'
        assertRaisesRegex(self, ValueError, expected_msg, OpenStackIdentity_3_0_Connection, auth_url='http://none', user_id='test', key='test', token_scope='foo')
        expected_msg = 'Must provide tenant_name and domain_name argument'
        assertRaisesRegex(self, ValueError, expected_msg, OpenStackIdentity_3_0_Connection, auth_url='http://none', user_id='test', key='test', token_scope='project')
        expected_msg = 'Must provide domain_name argument'
        assertRaisesRegex(self, ValueError, expected_msg, OpenStackIdentity_3_0_Connection, auth_url='http://none', user_id='test', key='test', token_scope='domain', domain_name=None)
        OpenStackIdentity_3_0_Connection(auth_url='http://none', user_id='test', key='test', token_scope='project', tenant_name='test', domain_name='Default')
        OpenStackIdentity_3_0_Connection(auth_url='http://none', user_id='test', key='test', token_scope='domain', tenant_name=None, domain_name='Default')

    def test_authenticate(self):
        auth = OpenStackIdentity_3_0_Connection(auth_url='http://none', user_id='test_user_id', key='test_key', token_scope='project', tenant_name='test_tenant', tenant_domain_id='test_tenant_domain_id', domain_name='test_domain', proxy_url='http://proxy:8080', timeout=10)
        auth.authenticate()
        self.assertEqual(auth.proxy_url, 'http://proxy:8080')

    def test_list_supported_versions(self):
        OpenStackIdentity_3_0_MockHttp.type = 'v3'
        versions = self.auth_instance.list_supported_versions()
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0].version, 'v2.0')
        self.assertEqual(versions[0].url, 'http://192.168.18.100:5000/v2.0/')
        self.assertEqual(versions[1].version, 'v3.0')
        self.assertEqual(versions[1].url, 'http://192.168.18.100:5000/v3/')

    def test_list_domains(self):
        domains = self.auth_instance.list_domains()
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, 'default')
        self.assertEqual(domains[0].name, 'Default')
        self.assertTrue(domains[0].enabled)

    def test_list_projects(self):
        projects = self.auth_instance.list_projects()
        self.assertEqual(len(projects), 4)
        self.assertEqual(projects[0].id, 'a')
        self.assertEqual(projects[0].domain_id, 'default')
        self.assertTrue(projects[0].enabled)
        self.assertEqual(projects[0].description, 'Test project')

    def test_list_users(self):
        users = self.auth_instance.list_users()
        self.assertEqual(len(users), 12)
        self.assertEqual(users[0].id, 'a')
        self.assertEqual(users[0].domain_id, 'default')
        self.assertEqual(users[0].enabled, True)
        self.assertEqual(users[0].email, 'openstack-test@localhost')

    def test_list_roles(self):
        roles = self.auth_instance.list_roles()
        self.assertEqual(len(roles), 2)
        self.assertEqual(roles[1].id, 'b')
        self.assertEqual(roles[1].name, 'admin')

    def test_list_user_projects(self):
        user = self.auth_instance.list_users()[0]
        projects = self.auth_instance.list_user_projects(user=user)
        self.assertEqual(len(projects), 0)

    def test_list_user_domain_roles(self):
        user = self.auth_instance.list_users()[0]
        domain = self.auth_instance.list_domains()[0]
        roles = self.auth_instance.list_user_domain_roles(domain=domain, user=user)
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].name, 'admin')

    def test_get_domain(self):
        domain = self.auth_instance.get_domain(domain_id='default')
        self.assertEqual(domain.name, 'Default')

    def test_get_user(self):
        user = self.auth_instance.get_user(user_id='a')
        self.assertEqual(user.id, 'a')
        self.assertEqual(user.domain_id, 'default')
        self.assertEqual(user.enabled, True)
        self.assertEqual(user.email, 'openstack-test@localhost')

    def test_get_user_without_email(self):
        user = self.auth_instance.get_user(user_id='b')
        self.assertEqual(user.id, 'b')
        self.assertEqual(user.name, 'userwithoutemail')
        self.assertIsNone(user.email)

    def test_get_user_without_enabled(self):
        user = self.auth_instance.get_user(user_id='c')
        self.assertEqual(user.id, 'c')
        self.assertEqual(user.name, 'userwithoutenabled')
        self.assertIsNone(user.enabled)

    def test_create_user(self):
        user = self.auth_instance.create_user(email='test2@localhost', password='test1', name='test2', domain_id='default')
        self.assertEqual(user.id, 'c')
        self.assertEqual(user.name, 'test2')

    def test_enable_user(self):
        user = self.auth_instance.list_users()[0]
        result = self.auth_instance.enable_user(user=user)
        self.assertTrue(isinstance(result, OpenStackIdentityUser))

    def test_disable_user(self):
        user = self.auth_instance.list_users()[0]
        result = self.auth_instance.disable_user(user=user)
        self.assertTrue(isinstance(result, OpenStackIdentityUser))

    def test_grant_domain_role_to_user(self):
        domain = self.auth_instance.list_domains()[0]
        role = self.auth_instance.list_roles()[0]
        user = self.auth_instance.list_users()[0]
        result = self.auth_instance.grant_domain_role_to_user(domain=domain, role=role, user=user)
        self.assertTrue(result)

    def test_revoke_domain_role_from_user(self):
        domain = self.auth_instance.list_domains()[0]
        role = self.auth_instance.list_roles()[0]
        user = self.auth_instance.list_users()[0]
        result = self.auth_instance.revoke_domain_role_from_user(domain=domain, role=role, user=user)
        self.assertTrue(result)

    def test_grant_project_role_to_user(self):
        project = self.auth_instance.list_projects()[0]
        role = self.auth_instance.list_roles()[0]
        user = self.auth_instance.list_users()[0]
        result = self.auth_instance.grant_project_role_to_user(project=project, role=role, user=user)
        self.assertTrue(result)

    def test_revoke_project_role_from_user(self):
        project = self.auth_instance.list_projects()[0]
        role = self.auth_instance.list_roles()[0]
        user = self.auth_instance.list_users()[0]
        result = self.auth_instance.revoke_project_role_from_user(project=project, role=role, user=user)
        self.assertTrue(result)