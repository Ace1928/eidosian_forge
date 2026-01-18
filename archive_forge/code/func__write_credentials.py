import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _write_credentials(self):
    if self._read_only:
        logger.debug('In read-only mode, not writing credentials.')
        return
    _write_credentials_file(self._file, self._credentials)
    logger.debug('Wrote credential file {0}.'.format(self._filename))