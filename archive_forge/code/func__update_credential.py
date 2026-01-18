import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _update_credential(self, key, cred):
    """Update a credential and write the multistore.

        This must be called when the multistore is locked.

        Args:
            key: The key used to retrieve the credential
            cred: The OAuth2Credential to update/set
        """
    self._data[key] = cred
    self._write()