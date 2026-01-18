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
class UnencryptedFileCredentialStore(CredentialStore):
    """Store credentials unencrypted in a file on disk.

    This is a good solution for scripts that need to run without any
    user interaction.
    """

    def __init__(self, filename, credential_save_failed=None):
        super(UnencryptedFileCredentialStore, self).__init__(credential_save_failed)
        self.filename = filename

    def do_save(self, credentials, unique_key):
        """Save the credentials to disk."""
        credentials.save_to_path(self.filename)

    def do_load(self, unique_key):
        """Load the credentials from disk."""
        if os.path.exists(self.filename) and (not os.stat(self.filename)[stat.ST_SIZE] == 0):
            return Credentials.load_from_path(self.filename)
        return None