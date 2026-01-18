import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _locked_json_read(self):
    """Get the raw content of the multistore file.

        The multistore must be locked when this is called.

        Returns:
            The contents of the multistore decoded as JSON.
        """
    assert self._thread_lock.locked()
    self._file.file_handle().seek(0)
    return json.load(self._file.file_handle())