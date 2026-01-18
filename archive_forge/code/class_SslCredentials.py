from __future__ import absolute_import
import logging
import os
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth.transport import _mtls_helper
from google.oauth2 import service_account
class SslCredentials:
    """Class for application default SSL credentials.

    The behavior is controlled by `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment
    variable whose default value is `false`. Client certificate will not be used
    unless the environment variable is explicitly set to `true`. See
    https://google.aip.dev/auth/4114

    If the environment variable is `true`, then for devices with endpoint verification
    support, a device certificate will be automatically loaded and mutual TLS will
    be established.
    See https://cloud.google.com/endpoint-verification/docs/overview.
    """

    def __init__(self):
        use_client_cert = os.getenv(environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE, 'false')
        if use_client_cert != 'true':
            self._is_mtls = False
        else:
            metadata_path = _mtls_helper._check_dca_metadata_path(_mtls_helper.CONTEXT_AWARE_METADATA_PATH)
            self._is_mtls = metadata_path is not None

    @property
    def ssl_credentials(self):
        """Get the created SSL channel credentials.

        For devices with endpoint verification support, if the device certificate
        loading has any problems, corresponding exceptions will be raised. For
        a device without endpoint verification support, no exceptions will be
        raised.

        Returns:
            grpc.ChannelCredentials: The created grpc channel credentials.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
                creation failed for any reason.
        """
        if self._is_mtls:
            try:
                _, cert, key, _ = _mtls_helper.get_client_ssl_credentials()
                self._ssl_credentials = grpc.ssl_channel_credentials(certificate_chain=cert, private_key=key)
            except exceptions.ClientCertError as caught_exc:
                new_exc = exceptions.MutualTLSChannelError(caught_exc)
                six.raise_from(new_exc, caught_exc)
        else:
            self._ssl_credentials = grpc.ssl_channel_credentials()
        return self._ssl_credentials

    @property
    def is_mtls(self):
        """Indicates if the created SSL channel credentials is mutual TLS."""
        return self._is_mtls