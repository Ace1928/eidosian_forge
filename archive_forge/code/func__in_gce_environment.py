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
def _in_gce_environment():
    """Detect if the code is running in the Compute Engine environment.

    Returns:
        True if running in the GCE environment, False otherwise.
    """
    if SETTINGS.env_name is not None:
        return SETTINGS.env_name == 'GCE_PRODUCTION'
    if NO_GCE_CHECK != 'True' and _detect_gce_environment():
        SETTINGS.env_name = 'GCE_PRODUCTION'
        return True
    return False