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
def _validate_start_and_end_bytes(self, start_bytes, end_bytes=None):
    """
        Method which validates that start_bytes and end_bytes arguments contain
        valid values.
        """
    if start_bytes < 0:
        raise ValueError('start_bytes must be greater than 0')
    if end_bytes is not None:
        if start_bytes > end_bytes:
            raise ValueError('start_bytes must be smaller than end_bytes')
        elif start_bytes == end_bytes:
            raise ValueError("start_bytes and end_bytes can't be the same. end_bytes is non-inclusive")
    return True