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
def _close_after_stream(response, chunk_size):
    """Iterate over the content and ensure the response is closed after."""
    for chunk in response.iter_content(chunk_size=chunk_size):
        yield chunk
    response.close()