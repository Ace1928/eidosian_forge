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
def _get_well_known_file():
    """Get the well known file produced by command 'gcloud auth login'."""
    default_config_dir = os.getenv(_CLOUDSDK_CONFIG_ENV_VAR)
    if default_config_dir is None:
        if os.name == 'nt':
            try:
                default_config_dir = os.path.join(os.environ['APPDATA'], _CLOUDSDK_CONFIG_DIRECTORY)
            except KeyError:
                drive = os.environ.get('SystemDrive', 'C:')
                default_config_dir = os.path.join(drive, '\\', _CLOUDSDK_CONFIG_DIRECTORY)
        else:
            default_config_dir = os.path.join(os.path.expanduser('~'), '.config', _CLOUDSDK_CONFIG_DIRECTORY)
    return os.path.join(default_config_dir, _WELL_KNOWN_CREDENTIALS_FILE)