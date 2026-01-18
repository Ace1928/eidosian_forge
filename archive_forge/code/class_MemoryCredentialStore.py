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
class MemoryCredentialStore(CredentialStore):
    """CredentialStore that stores keys only in memory.

    This can be used to provide a CredentialStore instance without
    actually saving any key to persistent storage.
    """

    def __init__(self, credential_save_failed=None):
        super(MemoryCredentialStore, self).__init__(credential_save_failed)
        self._credentials = {}

    def do_save(self, credentials, unique_key):
        """Store the credentials in our dict"""
        self._credentials[unique_key] = credentials

    def do_load(self, unique_key):
        """Retrieve the credentials from our dict"""
        return self._credentials.get(unique_key)