import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _load_credentials(self):
    """(Re-)loads the credentials from the file."""
    if not self._file:
        return
    loaded_credentials = _load_credentials_file(self._file)
    self._credentials.update(loaded_credentials)
    logger.debug('Read credential file')