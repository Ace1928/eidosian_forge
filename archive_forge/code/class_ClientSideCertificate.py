from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import inspect
import io
from google.auth.transport import requests as google_auth_requests
from google.auth.transport.requests import _MutualTlsOffloadAdapter
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import httplib2
import requests
import six
from six.moves import http_client as httplib
from six.moves import urllib
import socks
from urllib3.util.ssl_ import create_urllib3_context
class ClientSideCertificate(collections.namedtuple('ClientSideCertificate', ['certfile', 'keyfile', 'password'])):
    """Holds information about a client side certificate.

  Attributes:
    certfile: str, path to a cert file.
    keyfile: str, path to a key file.
    password: str, password to the private key.
  """

    def __new__(cls, certfile, keyfile, password=None):
        return super(ClientSideCertificate, cls).__new__(cls, certfile, keyfile, password)