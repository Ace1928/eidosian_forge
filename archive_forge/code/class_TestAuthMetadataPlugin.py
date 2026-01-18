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
class TestAuthMetadataPlugin(object):

    def test_call_no_refresh(self):
        credentials = CredentialsStub()
        request = mock.create_autospec(transport.Request)
        plugin = google.auth.transport.grpc.AuthMetadataPlugin(credentials, request)
        context = mock.create_autospec(grpc.AuthMetadataContext, instance=True)
        context.method_name = mock.sentinel.method_name
        context.service_url = mock.sentinel.service_url
        callback = mock.create_autospec(grpc.AuthMetadataPluginCallback)
        plugin(context, callback)
        time.sleep(2)
        callback.assert_called_once_with([('authorization', 'Bearer {}'.format(credentials.token))], None)

    def test_call_refresh(self):
        credentials = CredentialsStub()
        credentials.expiry = datetime.datetime.min + _helpers.REFRESH_THRESHOLD
        request = mock.create_autospec(transport.Request)
        plugin = google.auth.transport.grpc.AuthMetadataPlugin(credentials, request)
        context = mock.create_autospec(grpc.AuthMetadataContext, instance=True)
        context.method_name = mock.sentinel.method_name
        context.service_url = mock.sentinel.service_url
        callback = mock.create_autospec(grpc.AuthMetadataPluginCallback)
        plugin(context, callback)
        time.sleep(2)
        assert credentials.token == 'token1'
        callback.assert_called_once_with([('authorization', 'Bearer {}'.format(credentials.token))], None)

    def test__get_authorization_headers_with_service_account(self):
        credentials = mock.create_autospec(service_account.Credentials)
        request = mock.create_autospec(transport.Request)
        plugin = google.auth.transport.grpc.AuthMetadataPlugin(credentials, request)
        context = mock.create_autospec(grpc.AuthMetadataContext, instance=True)
        context.method_name = 'methodName'
        context.service_url = 'https://pubsub.googleapis.com/methodName'
        plugin._get_authorization_headers(context)
        credentials._create_self_signed_jwt.assert_called_once_with(None)

    def test__get_authorization_headers_with_service_account_and_default_host(self):
        credentials = mock.create_autospec(service_account.Credentials)
        request = mock.create_autospec(transport.Request)
        default_host = 'pubsub.googleapis.com'
        plugin = google.auth.transport.grpc.AuthMetadataPlugin(credentials, request, default_host=default_host)
        context = mock.create_autospec(grpc.AuthMetadataContext, instance=True)
        context.method_name = 'methodName'
        context.service_url = 'https://pubsub.googleapis.com/methodName'
        plugin._get_authorization_headers(context)
        credentials._create_self_signed_jwt.assert_called_once_with('https://{}/'.format(default_host))