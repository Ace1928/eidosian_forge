import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def _get_hash_function(self):
    """
        Return instantiated hash function for the hash type supported by
        the provider.
        """
    try:
        func = getattr(hashlib, self.hash_type)()
    except AttributeError:
        raise RuntimeError('Invalid or unsupported hash type: %s' % self.hash_type)
    return func