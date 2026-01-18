import copy
import io
import logging
import socket
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as ksa_exc
import OpenSSL
from oslo_utils import importutils
from oslo_utils import netutils
import requests
import urllib.parse
from oslo_utils import encodeutils
from glanceclient.common import utils
from glanceclient import exc
def _set_common_request_kwargs(self, headers, kwargs):
    """Handle the common parameters used to send the request."""
    content_type = headers.get('Content-Type', 'application/octet-stream')
    data = kwargs.pop('data', None)
    if data is not None and (not isinstance(data, str)):
        try:
            data = json.dumps(data)
            content_type = 'application/json'
        except TypeError:
            data = self._chunk_body(data)
    headers['Content-Type'] = content_type
    kwargs['stream'] = content_type == 'application/octet-stream'
    return data