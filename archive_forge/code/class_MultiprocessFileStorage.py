import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
class MultiprocessFileStorage(client.Storage):
    """Multiprocess file credential storage.

    Args:
      filename: The path to the file where credentials will be stored.
      key: An arbitrary string used to uniquely identify this set of
          credentials. For example, you may use the user's ID as the key or
          a combination of the client ID and user ID.
    """

    def __init__(self, filename, key):
        self._key = key
        self._backend = _get_backend(filename)

    def acquire_lock(self):
        self._backend.acquire_lock()

    def release_lock(self):
        self._backend.release_lock()

    def locked_get(self):
        """Retrieves the current credentials from the store.

        Returns:
            An instance of :class:`oauth2client.client.Credentials` or `None`.
        """
        credential = self._backend.locked_get(self._key)
        if credential is not None:
            credential.set_store(self)
        return credential

    def locked_put(self, credentials):
        """Writes the given credentials to the store.

        Args:
            credentials: an instance of
                :class:`oauth2client.client.Credentials`.
        """
        return self._backend.locked_put(self._key, credentials)

    def locked_delete(self):
        """Deletes the current credentials from the store."""
        return self._backend.locked_delete(self._key)