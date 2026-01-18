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