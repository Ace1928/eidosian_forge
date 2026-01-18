from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
class TestMapped(unit.TestCase):

    def config_files(self):
        config_files = super(TestMapped, self).config_files()
        config_files.append(unit.dirs.tests_conf('test_auth_plugin.conf'))
        return config_files

    def _test_mapped_invocation_with_method_name(self, method_name):
        with mock.patch.object(auth.plugins.mapped.Mapped, 'authenticate', return_value=None) as authenticate:
            auth_data = {'identity': {'methods': [method_name], method_name: {'protocol': method_name}}}
            auth_info = auth.core.AuthInfo.create(auth_data)
            auth_context = auth.core.AuthContext(method_names=[], user_id=uuid.uuid4().hex)
            with self.make_request():
                authentication.authenticate(auth_info, auth_context)
            (auth_payload,), kwargs = authenticate.call_args
            self.assertEqual(method_name, auth_payload['protocol'])

    def test_mapped_with_remote_user(self):
        method_name = 'saml2'
        auth_data = {'methods': [method_name]}
        auth_data[method_name] = {'protocol': method_name}
        auth_data = {'identity': auth_data}
        auth_context = auth.core.AuthContext(method_names=[], user_id=uuid.uuid4().hex)
        self.useFixture(auth_plugins.LoadAuthPlugins(method_name))
        with mock.patch.object(auth.plugins.mapped.Mapped, 'authenticate', return_value=None) as authenticate:
            auth_info = auth.core.AuthInfo.create(auth_data)
            with self.make_request(environ={'REMOTE_USER': 'foo@idp.com'}):
                authentication.authenticate(auth_info, auth_context)
            (auth_payload,), kwargs = authenticate.call_args
            self.assertEqual(method_name, auth_payload['protocol'])

    @mock.patch('keystone.auth.plugins.mapped.PROVIDERS')
    def test_mapped_without_identity_provider_or_protocol(self, mock_providers):
        mock_providers.resource_api = mock.Mock()
        mock_providers.federation_api = mock.Mock()
        mock_providers.identity_api = mock.Mock()
        mock_providers.assignment_api = mock.Mock()
        mock_providers.role_api = mock.Mock()
        test_mapped = mapped.Mapped()
        auth_payload = {'identity_provider': 'test_provider'}
        with self.make_request():
            self.assertRaises(exception.ValidationError, test_mapped.authenticate, auth_payload)
        auth_payload = {'protocol': 'saml2'}
        with self.make_request():
            self.assertRaises(exception.ValidationError, test_mapped.authenticate, auth_payload)

    def test_supporting_multiple_methods(self):
        method_names = ('saml2', 'openid', 'x509', 'mapped')
        self.useFixture(auth_plugins.LoadAuthPlugins(*method_names))
        for method_name in method_names:
            self._test_mapped_invocation_with_method_name(method_name)