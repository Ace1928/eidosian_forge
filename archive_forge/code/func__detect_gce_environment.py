import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _detect_gce_environment():
    """Determine if the current environment is Compute Engine.

    Returns:
        Boolean indicating whether or not the current environment is Google
        Compute Engine.
    """
    http = transport.get_http_object(timeout=GCE_METADATA_TIMEOUT)
    try:
        response, _ = transport.request(http, _GCE_METADATA_URI, headers=_GCE_HEADERS)
        return response.status == http_client.OK and response.get(_METADATA_FLAVOR_HEADER) == _DESIRED_METADATA_FLAVOR
    except socket.error:
        logger.info('Timeout attempting to reach GCE metadata service.')
        return False