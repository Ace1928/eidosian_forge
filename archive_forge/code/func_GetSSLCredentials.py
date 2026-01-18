from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from google.api_core import bidi
from google.rpc import error_details_pb2
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.calliope import base
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_proxy_types
import grpc
from six.moves import urllib
import socks
def GetSSLCredentials(mtls_enabled):
    """Returns SSL credentials."""
    ca_certs_file = properties.VALUES.core.custom_ca_certs_file.Get()
    certificate_chain = None
    private_key = None
    ca_config = context_aware.Config()
    if mtls_enabled and ca_config:
        log.debug('Using client certificate...')
        certificate_chain, private_key = (ca_config.client_cert_bytes, ca_config.client_key_bytes)
    if ca_certs_file or certificate_chain or private_key:
        if ca_certs_file:
            ca_certs = files.ReadBinaryFileContents(ca_certs_file)
        else:
            ca_certs = None
        return grpc.ssl_channel_credentials(root_certificates=ca_certs, certificate_chain=certificate_chain, private_key=private_key)
    return None