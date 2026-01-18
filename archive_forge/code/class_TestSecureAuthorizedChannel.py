import datetime
import os
import time
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.auth.transport._mtls_helper.get_client_ssl_credentials', autospec=True)
@mock.patch('grpc.composite_channel_credentials', autospec=True)
@mock.patch('grpc.metadata_call_credentials', autospec=True)
@mock.patch('grpc.ssl_channel_credentials', autospec=True)
@mock.patch('grpc.secure_channel', autospec=True)
class TestSecureAuthorizedChannel(object):

    @mock.patch('google.auth.transport._mtls_helper._read_dca_metadata_file', autospec=True)
    @mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
    def test_secure_authorized_channel_adc(self, check_dca_metadata_path, read_dca_metadata_file, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = CredentialsStub()
        request = mock.create_autospec(transport.Request)
        target = 'example.com:80'
        check_dca_metadata_path.return_value = METADATA_PATH
        read_dca_metadata_file.return_value = {'cert_provider_command': ['some command']}
        get_client_ssl_credentials.return_value = (True, PUBLIC_CERT_BYTES, PRIVATE_KEY_BYTES, None)
        channel = None
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            channel = google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, options=mock.sentinel.options)
        auth_plugin = metadata_call_credentials.call_args[0][0]
        assert isinstance(auth_plugin, google.auth.transport.grpc.AuthMetadataPlugin)
        assert auth_plugin._credentials == credentials
        assert auth_plugin._request == request
        ssl_channel_credentials.assert_called_once_with(certificate_chain=PUBLIC_CERT_BYTES, private_key=PRIVATE_KEY_BYTES)
        composite_channel_credentials.assert_called_once_with(ssl_channel_credentials.return_value, metadata_call_credentials.return_value)
        secure_channel.assert_called_once_with(target, composite_channel_credentials.return_value, options=mock.sentinel.options)
        assert channel == secure_channel.return_value

    @mock.patch('google.auth.transport.grpc.SslCredentials', autospec=True)
    def test_secure_authorized_channel_adc_without_client_cert_env(self, ssl_credentials_adc_method, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = CredentialsStub()
        request = mock.create_autospec(transport.Request)
        target = 'example.com:80'
        channel = google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, options=mock.sentinel.options)
        auth_plugin = metadata_call_credentials.call_args[0][0]
        assert isinstance(auth_plugin, google.auth.transport.grpc.AuthMetadataPlugin)
        assert auth_plugin._credentials == credentials
        assert auth_plugin._request == request
        ssl_channel_credentials.assert_called_once()
        ssl_credentials_adc_method.assert_not_called()
        composite_channel_credentials.assert_called_once_with(ssl_channel_credentials.return_value, metadata_call_credentials.return_value)
        secure_channel.assert_called_once_with(target, composite_channel_credentials.return_value, options=mock.sentinel.options)
        assert channel == secure_channel.return_value

    def test_secure_authorized_channel_explicit_ssl(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = mock.Mock()
        request = mock.Mock()
        target = 'example.com:80'
        ssl_credentials = mock.Mock()
        google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, ssl_credentials=ssl_credentials)
        assert not get_client_ssl_credentials.called
        assert not ssl_channel_credentials.called
        composite_channel_credentials.assert_called_once_with(ssl_credentials, metadata_call_credentials.return_value)

    def test_secure_authorized_channel_mutual_exclusive(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = mock.Mock()
        request = mock.Mock()
        target = 'example.com:80'
        ssl_credentials = mock.Mock()
        client_cert_callback = mock.Mock()
        with pytest.raises(ValueError):
            google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, ssl_credentials=ssl_credentials, client_cert_callback=client_cert_callback)

    def test_secure_authorized_channel_with_client_cert_callback_success(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = mock.Mock()
        request = mock.Mock()
        target = 'example.com:80'
        client_cert_callback = mock.Mock()
        client_cert_callback.return_value = (PUBLIC_CERT_BYTES, PRIVATE_KEY_BYTES)
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, client_cert_callback=client_cert_callback)
        client_cert_callback.assert_called_once()
        ssl_channel_credentials.assert_called_once_with(certificate_chain=PUBLIC_CERT_BYTES, private_key=PRIVATE_KEY_BYTES)
        composite_channel_credentials.assert_called_once_with(ssl_channel_credentials.return_value, metadata_call_credentials.return_value)

    @mock.patch('google.auth.transport._mtls_helper._read_dca_metadata_file', autospec=True)
    @mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
    def test_secure_authorized_channel_with_client_cert_callback_failure(self, check_dca_metadata_path, read_dca_metadata_file, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = mock.Mock()
        request = mock.Mock()
        target = 'example.com:80'
        client_cert_callback = mock.Mock()
        client_cert_callback.side_effect = Exception('callback exception')
        with pytest.raises(Exception) as excinfo:
            with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
                google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, client_cert_callback=client_cert_callback)
        assert str(excinfo.value) == 'callback exception'

    def test_secure_authorized_channel_cert_callback_without_client_cert_env(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
        credentials = mock.Mock()
        request = mock.Mock()
        target = 'example.com:80'
        client_cert_callback = mock.Mock()
        google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, client_cert_callback=client_cert_callback)
        client_cert_callback.assert_not_called()
        ssl_channel_credentials.assert_called_once()
        composite_channel_credentials.assert_called_once_with(ssl_channel_credentials.return_value, metadata_call_credentials.return_value)