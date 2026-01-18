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
@staticmethod
def _implicit_credentials_from_gae():
    """Attempts to get implicit credentials in Google App Engine env.

        If the current environment is not detected as App Engine, returns None,
        indicating no Google App Engine credentials can be detected from the
        current environment.

        Returns:
            None, if not in GAE, else an appengine.AppAssertionCredentials
            object.
        """
    if not _in_gae_environment():
        return None
    return _get_application_default_credential_GAE()