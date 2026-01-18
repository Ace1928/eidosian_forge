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
def _delete_empty_container(conn, container, options):
    results_dict = {}
    _headers = {}
    _headers = split_headers(options.get('header', []))
    try:
        conn.delete_container(container, headers=_headers, response_dict=results_dict)
        res = {'success': True}
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        res = {'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time}
    res.update({'action': 'delete_container', 'container': container, 'object': None, 'attempts': conn.attempts, 'response_dict': results_dict})
    return res