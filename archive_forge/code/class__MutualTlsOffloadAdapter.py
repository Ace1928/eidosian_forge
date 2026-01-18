from __future__ import absolute_import
import functools
import logging
import numbers
import os
import time
import requests.adapters  # pylint: disable=ungrouped-imports
import requests.exceptions  # pylint: disable=ungrouped-imports
from requests.packages.urllib3.util.ssl_ import (  # type: ignore
import six  # pylint: disable=ungrouped-imports
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
import google.auth.transport._mtls_helper
from google.oauth2 import service_account
class _MutualTlsOffloadAdapter(requests.adapters.HTTPAdapter):
    """
    A TransportAdapter that enables mutual TLS and offloads the client side
    signing operation to the signing library.

    Args:
        enterprise_cert_file_path (str): the path to a enterprise cert JSON
            file. The file should contain the following field:

                {
                    "libs": {
                        "signer_library": "...",
                        "offload_library": "..."
                    }
                }

    Raises:
        ImportError: if certifi or pyOpenSSL is not installed
        google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
            creation failed for any reason.
    """

    def __init__(self, enterprise_cert_file_path):
        import certifi
        import urllib3.contrib.pyopenssl
        from google.auth.transport import _custom_tls_signer
        urllib3.contrib.pyopenssl.inject_into_urllib3()
        self.signer = _custom_tls_signer.CustomTlsSigner(enterprise_cert_file_path)
        self.signer.load_libraries()
        self.signer.set_up_custom_key()
        poolmanager = create_urllib3_context()
        poolmanager.load_verify_locations(cafile=certifi.where())
        self.signer.attach_to_ssl_context(poolmanager)
        self._ctx_poolmanager = poolmanager
        proxymanager = create_urllib3_context()
        proxymanager.load_verify_locations(cafile=certifi.where())
        self.signer.attach_to_ssl_context(proxymanager)
        self._ctx_proxymanager = proxymanager
        super(_MutualTlsOffloadAdapter, self).__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self._ctx_poolmanager
        super(_MutualTlsOffloadAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = self._ctx_proxymanager
        return super(_MutualTlsOffloadAdapter, self).proxy_manager_for(*args, **kwargs)