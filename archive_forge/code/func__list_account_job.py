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
def _list_account_job(conn, options, result_queue):
    marker = ''
    error = None
    req_headers = split_headers(options.get('header', []))
    try:
        while True:
            _, items = conn.get_account(marker=marker, prefix=options['prefix'], headers=req_headers)
            if not items:
                result_queue.put(None)
                return
            if options['long']:
                for i in items:
                    name = i['name']
                    i['meta'] = conn.head_container(name)
            res = {'action': 'list_account_part', 'container': None, 'prefix': options['prefix'], 'success': True, 'listing': items, 'marker': marker}
            result_queue.put(res)
            marker = items[-1].get('name', items[-1].get('subdir'))
    except ClientException as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        if err.http_status != 404:
            error = (err, traceback, err_time)
        else:
            error = (SwiftError('Account not found', exc=err), traceback, err_time)
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        error = (err, traceback, err_time)
    res = {'action': 'list_account_part', 'container': None, 'prefix': options['prefix'], 'success': False, 'marker': marker, 'error': error[0], 'traceback': error[1], 'error_timestamp': error[2]}
    result_queue.put(res)
    result_queue.put(None)