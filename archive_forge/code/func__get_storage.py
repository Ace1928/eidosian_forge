import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _get_storage(self, key):
    """Get a Storage object to get/set a credential.

        This Storage is a 'view' into the multistore.

        Args:
            key: The key used to retrieve the credential

        Returns:
            A Storage object that can be used to get/set this cred
        """
    return self._Storage(self, key)