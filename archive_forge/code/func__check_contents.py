import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
def _check_contents(self):
    if self._content_length is not None and self._actual_read != self._content_length:
        raise SwiftError('Error downloading {0}: read_length != content_length, {1:d} != {2:d} (txn: {3})'.format(self._path, self._actual_read, self._content_length, self._txn_id or 'unknown'))
    if self._actual_md5 and self._expected_md5:
        etag = self._actual_md5.hexdigest()
        if etag != self._expected_md5:
            raise SwiftError('Error downloading {0}: md5sum != etag, {1} != {2} (txn: {3})'.format(self._path, etag, self._expected_md5, self._txn_id or 'unknown'))