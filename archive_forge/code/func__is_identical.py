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
def _is_identical(self, chunk_data, path):
    if path is None:
        return False
    try:
        fp = open(path, 'rb', DISK_BUFFER)
    except IOError:
        return False
    with fp:
        for chunk in chunk_data:
            to_read = chunk['bytes']
            md5sum = md5()
            while to_read:
                data = fp.read(min(DISK_BUFFER, to_read))
                if not data:
                    return False
                md5sum.update(data)
                to_read -= len(data)
            if md5sum.hexdigest() != chunk['hash']:
                return False
        return not fp.read(1)