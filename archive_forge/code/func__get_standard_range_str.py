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
def _get_standard_range_str(self, start_bytes, end_bytes=None, end_bytes_inclusive=False):
    """
        Return range string which is used as a Range header value for range
        requests for drivers which follow standard Range header notation

        This returns range string in the following format:
        bytes=<start_bytes>-<end bytes>.

        For example:

        bytes=1-10
        bytes=0-2
        bytes=5-
        bytes=100-5000

        :param end_bytes_inclusive: True if "end_bytes" offset should be
        inclusive (aka opposite from the Python indexing behavior where the end
        index is not inclusive).
        """
    range_str = 'bytes=%s-' % start_bytes
    if end_bytes is not None:
        if end_bytes_inclusive:
            range_str += str(end_bytes)
        else:
            range_str += str(end_bytes - 1)
    return range_str