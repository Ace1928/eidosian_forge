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
@staticmethod
def _post_container_job(conn, container, headers, result):
    try:
        res = conn.post_container(container, headers=headers, response_dict=result)
    except ClientException as err:
        if err.http_status != 404:
            raise
        _response_dict = {}
        res = conn.put_container(container, headers=headers, response_dict=_response_dict)
        result['post_put'] = _response_dict
    return res