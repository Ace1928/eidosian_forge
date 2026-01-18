from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import base64
import contextlib
import os
import re
import ssl
import sys
import tempfile
from googlecloudsdk.api_lib.run import gke
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import files
import requests
import six
from six.moves.urllib import parse as urlparse
class KubeconfigConnectionContext(ConnectionInfo):
    """Context manager to connect to a cluster defined in a Kubeconfig file."""

    def __init__(self, kubeconfig, api_name, version, context=None):
        """Initialize connection context based on kubeconfig file.

    Args:
      kubeconfig: googlecloudsdk.api_lib.container.kubeconfig.Kubeconfig object
      api_name: str, api name to use for making requests
      version: str, api version to use for making requests
      context: str, current context name
    """
        super(KubeconfigConnectionContext, self).__init__(api_name, version)
        self.kubeconfig = kubeconfig
        self.kubeconfig.SetCurrentContext(context or kubeconfig.current_context)
        self.client_cert_data = None
        self.client_cert = None
        self.client_key = None
        self.client_cert_domain = None

    @contextlib.contextmanager
    def Connect(self):
        _CheckTLSSupport()
        with self._LoadClusterDetails():
            try:
                if self.ca_data:
                    with gke.MonkeypatchAddressChecking('kubernetes.default', self.raw_hostname) as endpoint:
                        self.endpoint = 'https://{}/{}'.format(endpoint, self.raw_path)
                else:
                    self.endpoint = 'https://{}/{}'.format(self.raw_hostname, self.raw_path)
                with _OverrideEndpointOverrides(self._api_name, self.endpoint):
                    yield self
            except (ssl.SSLError, requests.exceptions.SSLError) as e:
                if 'CERTIFICATE_VERIFY_FAILED' in six.text_type(e):
                    raise gke.NoCaCertError('Missing or invalid [certificate-authority] or [certificate-authority-data] field in kubeconfig file.')
                else:
                    raise

    def HttpClient(self):
        assert self.active
        if not self.client_key and self.client_cert and self.client_cert_domain:
            raise ValueError('Kubeconfig authentication requires a client certificate authentication method.')
        if self.client_cert_domain:
            from googlecloudsdk.core import transports
            http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, ca_certs=self.ca_certs, client_certificate=self.client_cert, client_key=self.client_key, client_cert_domain=self.client_cert_domain)
            return http_client
        from googlecloudsdk.core.credentials import transports
        http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, ca_certs=self.ca_certs)
        return http_client

    @property
    def operator(self):
        return 'Cloud Run for Anthos'

    @property
    def location_label(self):
        return ' of context [{{{{bold}}}}{}{{{{reset}}}}] referenced by config file [{{{{bold}}}}{}{{{{reset}}}}]'.format(self.curr_ctx['name'], self.kubeconfig.filename)

    @property
    def cluster_name(self):
        return self.cluster['name']

    @property
    def cluster_location(self):
        return None

    @property
    def supports_one_platform(self):
        return False

    @property
    def ns_label(self):
        return 'namespace'

    @contextlib.contextmanager
    def _WriteDataIfNoFile(self, f, d):
        if f:
            yield f
        elif d:
            fd, f = tempfile.mkstemp()
            os.close(fd)
            try:
                files.WriteBinaryFileContents(f, base64.b64decode(d), private=True)
                yield f
            finally:
                os.remove(f)
        else:
            yield None

    @contextlib.contextmanager
    def _LoadClusterDetails(self):
        """Get the current cluster and its connection info from the kubeconfig.

    Yields:
      None.
    Raises:
      flags.KubeconfigError: if the config file has missing keys or values.
    """
        try:
            self.curr_ctx = self.kubeconfig.contexts[self.kubeconfig.current_context]
            self.cluster = self.kubeconfig.clusters[self.curr_ctx['context']['cluster']]
            self.ca_certs = self.cluster['cluster'].get('certificate-authority', None)
            if not self.ca_certs:
                self.ca_data = self.cluster['cluster'].get('certificate-authority-data', None)
            parsed_server = urlparse.urlparse(self.cluster['cluster']['server'])
            self.raw_hostname = parsed_server.hostname
            if parsed_server.path:
                self.raw_path = parsed_server.path.strip('/') + '/'
            else:
                self.raw_path = ''
            self.user = self.kubeconfig.users[self.curr_ctx['context']['user']]
            self.client_key = self.user['user'].get('client-key', None)
            self.client_key_data = None
            self.client_cert_data = None
            if not self.client_key:
                self.client_key_data = self.user['user'].get('client-key-data', None)
            self.client_cert = self.user['user'].get('client-certificate', None)
            if not self.client_cert:
                self.client_cert_data = self.user['user'].get('client-certificate-data', None)
        except KeyError as e:
            raise flags.KubeconfigError('Missing key `{}` in kubeconfig.'.format(e.args[0]))
        with self._WriteDataIfNoFile(self.ca_certs, self.ca_data) as ca_certs, self._WriteDataIfNoFile(self.client_key, self.client_key_data) as client_key, self._WriteDataIfNoFile(self.client_cert, self.client_cert_data) as client_cert:
            self.ca_certs = ca_certs
            self.client_key = client_key
            self.client_cert = client_cert
            if self.client_cert:
                if six.PY2:
                    self.client_cert_domain = 'kubernetes.default'
                else:
                    self.client_cert_domain = self.raw_hostname
            yield