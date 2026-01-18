from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
class OAuth2CertificateTests(test_v3.OAuth2RestfulTestCase):
    ACCESS_TOKEN_URL = '/OS-OAUTH2/token'

    def setUp(self):
        super(OAuth2CertificateTests, self).setUp()
        self.log_fix = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        self.config_fixture.config(group='oauth2', oauth2_authn_methods=['tls_client_auth'])
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_mapping')
        self.oauth2_user, self.oauth2_user_domain, _ = self._create_project_user()
        *_, self.client_cert, self.client_key = self._create_certificates(client_dn=unit.create_dn(user_id=self.oauth2_user.get('id'), common_name=self.oauth2_user.get('name'), email_address=self.oauth2_user.get('email'), domain_component=self.oauth2_user_domain.get('id'), organization_name=self.oauth2_user_domain.get('name')))

    def _create_project_user(self, no_roles=False):
        new_domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain_ref['id'], new_domain_ref)
        new_project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(new_project_ref['id'], new_project_ref)
        new_user = unit.create_user(PROVIDERS.identity_api, domain_id=new_domain_ref['id'], project_id=new_project_ref['id'])
        if not no_roles:
            PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=new_user['id'], project_id=new_project_ref['id'])
        return (new_user, new_domain_ref, new_project_ref)

    def _create_certificates(self, root_dn=None, server_dn=None, client_dn=None):
        root_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='root')
        if root_dn:
            root_subj = unit.update_dn(root_subj, root_dn)
        root_cert, root_key = unit.create_certificate(root_subj)
        keystone_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='keystone.local')
        if server_dn:
            keystone_subj = unit.update_dn(keystone_subj, server_dn)
        ks_cert, ks_key = unit.create_certificate(keystone_subj, ca=root_cert, ca_key=root_key)
        client_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test')
        if client_dn:
            client_subj = unit.update_dn(client_subj, client_dn)
        client_cert, client_key = unit.create_certificate(client_subj, ca=root_cert, ca_key=root_key)
        return (root_cert, root_key, ks_cert, ks_key, client_cert, client_key)

    def _create_mapping(self, id='oauth2_mapping', dn_rules=None):
        rules = []
        if not dn_rules:
            dn_rules = [{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.email': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}]
        for info in dn_rules:
            index = 0
            local_user = {}
            remote = []
            for k in info:
                if k == 'user.name':
                    local_user['name'] = '{%s}' % index
                    remote.append({'type': info.get(k)})
                    index += 1
                elif k == 'user.id':
                    local_user['id'] = '{%s}' % index
                    remote.append({'type': info.get(k)})
                    index += 1
                elif k == 'user.email':
                    local_user['email'] = '{%s}' % index
                    remote.append({'type': info.get(k)})
                    index += 1
                elif k == 'user.domain.name' or k == 'user.domain.id':
                    if not local_user.get('domain'):
                        local_user['domain'] = {}
                    if k == 'user.domain.name':
                        local_user['domain']['name'] = '{%s}' % index
                        remote.append({'type': info.get(k)})
                        index += 1
                    else:
                        local_user['domain']['id'] = '{%s}' % index
                        remote.append({'type': info.get(k)})
                        index += 1
                else:
                    remote.append({'type': k, 'any_one_of': info.get(k)})
            rule = {'local': [{'user': local_user}], 'remote': remote}
            rules.append(rule)
        mapping = {'id': id, 'rules': rules}
        PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)

    def _get_access_token(self, client_id=None, client_cert_content=None, expected_status=http.client.OK):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials'}
        if client_id:
            data.update({'client_id': client_id})
        data = parse.urlencode(data).encode()
        kwargs = {'headers': headers, 'noauth': True, 'convert': False, 'body': data, 'expected_status': expected_status}
        if client_cert_content:
            kwargs.update({'environ': {'SSL_CLIENT_CERT': client_cert_content}})
        resp = self.post(self.ACCESS_TOKEN_URL, **kwargs)
        return resp

    def _get_cert_content(self, cert):
        return cert.public_bytes(Encoding.PEM).decode('ascii')

    def assertUnauthorizedResp(self, resp):
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertEqual('invalid_client', json_resp['error'])
        self.assertEqual('Client authentication failed.', json_resp['error_description'])

    def test_get_access_token_project_scope(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping()
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_mapping_config(self):
        """Test case when an access token can be successfully obtain."""
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_custom')
        self._create_mapping(id='oauth2_custom', dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_DC', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id='test_UID', common_name=user.get('name'), domain_component=user_domain.get('name'), organization_name='test_O'))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_mapping')

    def test_get_access_token_mapping_multi_ca(self):
        """Test case when an access token can be successfully obtain."""
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_custom')
        self._create_mapping(id='oauth2_custom', dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.email': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['rootA', 'rootB']}, {'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_DC', 'SSL_CLIENT_ISSUER_DN_CN': ['rootC']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(root_dn=unit.create_dn(common_name='rootA'), client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(root_dn=unit.create_dn(common_name='rootB'), client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(root_dn=unit.create_dn(common_name='rootC'), client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id='test_UID', common_name=user.get('name'), domain_component=user_domain.get('name'), organization_name='test_O'))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(root_dn=unit.create_dn(common_name='root_other'), client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_mapping')

    def test_get_access_token_no_default_mapping(self):
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping id %s is not found. ' % 'oauth2_mapping', self.log_fix.output)

    def test_get_access_token_no_custom_mapping(self):
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_custom')
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping id %s is not found. ' % 'oauth2_custom', self.log_fix.output)
        self.config_fixture.config(group='oauth2', oauth2_cert_dn_mapping_id='oauth2_mapping')

    def test_get_access_token_ignore_userid(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping(dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.email': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id') + '_diff', common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_ignore_username(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping(dn_rules=[{'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.email': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_ignore_email(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping(dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_ignore_domain_id(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping(dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.email': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id') + '_diff', organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_ignore_domain_name(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping(dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.email': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_ignore_all(self):
        """Test case when an access token can be successfully obtain."""
        self._create_mapping(dn_rules=[{'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
        user, user_domain, user_project = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id') + '_diff', common_name=user.get('name') + '_diff', email_address=user.get('email') + '_diff', domain_component=user_domain.get('id') + '_diff'))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])

    def test_get_access_token_no_roles_project_scope(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user(no_roles=True)
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        LOG.debug(resp)

    def test_get_access_token_no_default_project_id(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user(no_roles=True)
        user['default_project_id'] = None
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        _ = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)

    def test_get_access_token_without_client_id(self):
        self._create_mapping()
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: failed to get a client_id from the request.', self.log_fix.output)

    def test_get_access_token_without_client_cert(self):
        self._create_mapping()
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: failed to get client credentials from the request.', self.log_fix.output)

    @mock.patch.object(utils, 'get_certificate_subject_dn')
    def test_get_access_token_failed_to_get_cert_subject_dn(self, mock_get_certificate_subject_dn):
        self._create_mapping()
        mock_get_certificate_subject_dn.side_effect = exception.ValidationError('Boom!')
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: failed to get the subject DN from the certificate.', self.log_fix.output)

    @mock.patch.object(utils, 'get_certificate_issuer_dn')
    def test_get_access_token_failed_to_get_cert_issuer_dn(self, mock_get_certificate_issuer_dn):
        self._create_mapping()
        mock_get_certificate_issuer_dn.side_effect = exception.ValidationError('Boom!')
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: failed to get the issuer DN from the certificate.', self.log_fix.output)

    def test_get_access_token_user_not_exist(self):
        self._create_mapping()
        cert_content = self._get_cert_content(self.client_cert)
        user_id_not_exist = 'user_id_not_exist'
        resp = self._get_access_token(client_id=user_id_not_exist, client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: the user does not exist. user id: %s' % user_id_not_exist, self.log_fix.output)

    def test_get_access_token_cert_dn_not_match_user_id(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id') + '_diff', common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.' % ('user id', user.get('id') + '_diff', user.get('id')), self.log_fix.output)

    def test_get_access_token_cert_dn_not_match_user_name(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name') + '_diff', email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.' % ('user name', user.get('name') + '_diff', user.get('name')), self.log_fix.output)

    def test_get_access_token_cert_dn_not_match_email(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email') + '_diff', domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.' % ('user email', user.get('email') + '_diff', user.get('email')), self.log_fix.output)

    def test_get_access_token_cert_dn_not_match_domain_id(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id') + '_diff', organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.' % ('user domain id', user_domain.get('id') + '_diff', user_domain.get('id')), self.log_fix.output)

    def test_get_access_token_cert_dn_not_match_domain_name(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name') + '_diff'))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.' % ('user domain name', user_domain.get('name') + '_diff', user_domain.get('name')), self.log_fix.output)

    def test_get_access_token_cert_dn_missing_user_id(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)

    def test_get_access_token_cert_dn_missing_user_name(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), email_address=user.get('email'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)

    def test_get_access_token_cert_dn_missing_email(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)

    def test_get_access_token_cert_dn_missing_domain_id(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), organization_name=user_domain.get('name')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)

    def test_get_access_token_cert_dn_missing_domain_name(self):
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id')))
        cert_content = self._get_cert_content(client_cert)
        resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        self.assertUnauthorizedResp(resp)
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)

    @mock.patch.object(Manager, 'issue_token')
    def test_get_access_token_issue_token_ks_error_400(self, mock_issue_token):
        self._create_mapping()
        err_msg = 'Boom!'
        mock_issue_token.side_effect = exception.ValidationError(err_msg)
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.BAD_REQUEST)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertEqual('invalid_request', json_resp['error'])
        self.assertEqual(err_msg, json_resp['error_description'])
        self.assertIn(err_msg, self.log_fix.output)

    @mock.patch.object(Manager, 'issue_token')
    def test_get_access_token_issue_token_ks_error_401(self, mock_issue_token):
        self._create_mapping()
        err_msg = 'Boom!'
        mock_issue_token.side_effect = exception.Unauthorized(err_msg)
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertEqual('invalid_client', json_resp['error'])
        self.assertEqual('The request you have made requires authentication.', json_resp['error_description'])

    @mock.patch.object(Manager, 'issue_token')
    def test_get_access_token_issue_token_ks_error_other(self, mock_issue_token):
        self._create_mapping()
        err_msg = 'Boom!'
        mock_issue_token.side_effect = exception.NotImplemented(err_msg)
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=exception.NotImplemented.code)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertEqual('other_error', json_resp['error'])
        self.assertEqual('An unknown error occurred and failed to get an OAuth2.0 access token.', json_resp['error_description'])

    @mock.patch.object(Manager, 'issue_token')
    def test_get_access_token_issue_token_other_exception(self, mock_issue_token):
        self._create_mapping()
        err_msg = 'Boom!'
        mock_issue_token.side_effect = Exception(err_msg)
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.INTERNAL_SERVER_ERROR)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertEqual('other_error', json_resp['error'])
        self.assertEqual(err_msg, json_resp['error_description'])

    @mock.patch.object(RuleProcessor, 'process')
    def test_get_access_token_process_other_exception(self, mock_process):
        self._create_mapping()
        err_msg = 'Boom!'
        mock_process.side_effect = Exception(err_msg)
        cert_content = self._get_cert_content(self.client_cert)
        resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.INTERNAL_SERVER_ERROR)
        LOG.debug(resp)
        json_resp = jsonutils.loads(resp.body)
        self.assertEqual('other_error', json_resp['error'])
        self.assertEqual(err_msg, json_resp['error_description'])
        self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)