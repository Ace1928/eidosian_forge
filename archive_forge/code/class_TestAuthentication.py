import os
from urllib import parse
import tempest.lib.cli.base
from novaclient import client
from novaclient.tests.functional import base
class TestAuthentication(base.ClientTestBase):

    def _get_url(self, identity_api_version):
        url = parse.urlparse(self.cli_clients.uri)
        return parse.urlunparse((url.scheme, url.netloc, '/identity/v%s' % identity_api_version, url.params, url.query, url.fragment))

    def nova_auth_with_password(self, action, identity_api_version):
        flags = f'--os-username {self.cli_clients.username} --os-tenant-name {self.cli_clients.tenant_name} --os-password {self.cli_clients.password} --os-auth-url {self._get_url(identity_api_version)} --os-endpoint-type publicURL'
        if self.cacert:
            flags = f'{flags} --os-cacert {self.cacert}'
        if self.cert:
            flags = f'{flags} --os-cert {self.cert}'
        if self.cli_clients.insecure:
            flags = f'{flags} --insecure'
        return tempest.lib.cli.base.execute('nova', action, flags, cli_dir=self.cli_clients.cli_dir)

    def nova_auth_with_token(self, identity_api_version):
        auth_ref = self.client.client.session.auth.get_access(self.client.client.session)
        token = auth_ref.auth_token
        auth_url = self._get_url(identity_api_version)
        kw = {}
        if identity_api_version == '3':
            kw['project_domain_id'] = self.project_domain_id
        nova = client.Client('2', auth_token=token, auth_url=auth_url, project_name=self.project_name, cacert=self.cacert, cert=self.cert, **kw)
        nova.servers.list()
        os.environ.pop('OS_AUTH_TYPE', None)
        os.environ.pop('OS_AUTH_PLUGIN', None)
        flags = f'--os-tenant-name {self.project_name} --os-token {token} --os-auth-url {auth_url} --os-endpoint-type publicURL'
        if self.cacert:
            flags = f'{flags} --os-cacert {self.cacert}'
        if self.cert:
            flags = f'{flags} --os-cert {self.cert}'
        if self.cli_clients.insecure:
            flags = f'{flags} --insecure'
        tempest.lib.cli.base.execute('nova', 'list', flags, cli_dir=self.cli_clients.cli_dir)

    def test_auth_via_keystone_v2(self):
        session = self.keystone.session
        version = (2, 0)
        if not base.is_keystone_version_available(session, version):
            self.skipTest('Identity API version 2.0 is not available.')
        self.nova_auth_with_password('list', identity_api_version='2.0')
        self.nova_auth_with_token(identity_api_version='2.0')

    def test_auth_via_keystone_v3(self):
        session = self.keystone.session
        version = (3, 0)
        if not base.is_keystone_version_available(session, version):
            self.skipTest('Identity API version 3.0 is not available.')
        self.nova_auth_with_password('list', identity_api_version='3')
        self.nova_auth_with_token(identity_api_version='3')