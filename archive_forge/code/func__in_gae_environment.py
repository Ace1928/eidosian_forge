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
def _in_gae_environment():
    """Detects if the code is running in the App Engine environment.

    Returns:
        True if running in the GAE environment, False otherwise.
    """
    if SETTINGS.env_name is not None:
        return SETTINGS.env_name in ('GAE_PRODUCTION', 'GAE_LOCAL')
    try:
        import google.appengine
    except ImportError:
        pass
    else:
        server_software = os.environ.get(_SERVER_SOFTWARE, '')
        if server_software.startswith('Google App Engine/'):
            SETTINGS.env_name = 'GAE_PRODUCTION'
            return True
        elif server_software.startswith('Development/'):
            SETTINGS.env_name = 'GAE_LOCAL'
            return True
    return False