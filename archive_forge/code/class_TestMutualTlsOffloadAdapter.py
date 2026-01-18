import datetime
import functools
import os
import sys
import freezegun
import mock
import OpenSSL
import pytest  # type: ignore
import requests
import requests.adapters
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._custom_tls_signer
import google.auth.transport._mtls_helper
import google.auth.transport.requests
from google.oauth2 import service_account
from tests.transport import compliance
class TestMutualTlsOffloadAdapter(object):

    @mock.patch.object(requests.adapters.HTTPAdapter, 'init_poolmanager')
    @mock.patch.object(requests.adapters.HTTPAdapter, 'proxy_manager_for')
    @mock.patch.object(google.auth.transport._custom_tls_signer.CustomTlsSigner, 'load_libraries')
    @mock.patch.object(google.auth.transport._custom_tls_signer.CustomTlsSigner, 'set_up_custom_key')
    @mock.patch.object(google.auth.transport._custom_tls_signer.CustomTlsSigner, 'attach_to_ssl_context')
    def test_success(self, mock_attach_to_ssl_context, mock_set_up_custom_key, mock_load_libraries, mock_proxy_manager_for, mock_init_poolmanager):
        enterprise_cert_file_path = '/path/to/enterprise/cert/json'
        adapter = google.auth.transport.requests._MutualTlsOffloadAdapter(enterprise_cert_file_path)
        mock_load_libraries.assert_called_once()
        mock_set_up_custom_key.assert_called_once()
        assert mock_attach_to_ssl_context.call_count == 2
        adapter.init_poolmanager()
        mock_init_poolmanager.assert_called_with(ssl_context=adapter._ctx_poolmanager)
        adapter.proxy_manager_for()
        mock_proxy_manager_for.assert_called_with(ssl_context=adapter._ctx_proxymanager)