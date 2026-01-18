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
class HttpAccessTokenRefreshError(AccessTokenRefreshError):
    """Error (with HTTP status) trying to refresh an expired access token."""

    def __init__(self, *args, **kwargs):
        super(HttpAccessTokenRefreshError, self).__init__(*args)
        self.status = kwargs.get('status')