import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _create_file_if_needed(filename):
    """Creates the an empty file if it does not already exist.

    Returns:
        True if the file was created, False otherwise.
    """
    if os.path.exists(filename):
        return False
    else:
        old_umask = os.umask(127)
        try:
            open(filename, 'a+b').close()
        finally:
            os.umask(old_umask)
        logger.info('Credential file {0} created'.format(filename))
        return True