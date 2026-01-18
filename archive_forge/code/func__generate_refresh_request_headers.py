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
def _generate_refresh_request_headers(self):
    """Generate the headers that will be used in the refresh request."""
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    if self.user_agent is not None:
        headers['user-agent'] = self.user_agent
    return headers