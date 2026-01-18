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
def _create_container_job(conn, container, headers=None, policy_source=None):
    """
        Create a container using the given connection

        :param conn: The swift connection used for requests.
        :param container: The container name to create.
        :param headers: An optional dict of headers for the
                        put_container request.
        :param policy_source: An optional name of a container whose policy we
                              should duplicate.
        :return: A dict containing the results of the operation.
        """
    res = {'action': 'create_container', 'container': container, 'headers': headers}
    create_response = {}
    try:
        if policy_source is not None:
            _meta = conn.head_container(policy_source)
            if 'x-storage-policy' in _meta:
                policy_header = {POLICY: _meta.get('x-storage-policy')}
                if headers is None:
                    headers = policy_header
                else:
                    headers.update(policy_header)
        conn.put_container(container, headers, response_dict=create_response)
        res.update({'success': True, 'response_dict': create_response})
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': create_response})
    return res