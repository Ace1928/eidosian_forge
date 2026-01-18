from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
class AnonymousAccessToken(_AccessToken):
    """An OAuth access token that doesn't authenticate anybody.

    This token can be used for anonymous access.
    """

    def __init__(self):
        super(AnonymousAccessToken, self).__init__('', '')