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
def _get_chunk_data(self, conn, container, obj, headers, manifest=None):
    chunks = []
    if 'x-object-manifest' in headers:
        scontainer, sprefix = headers['x-object-manifest'].split('/', 1)
        for part in self.list(scontainer, {'prefix': sprefix}):
            if part['success']:
                chunks.extend(part['listing'])
            else:
                raise part['error']
    elif config_true_value(headers.get('x-static-large-object')):
        if manifest is None:
            headers, manifest = conn.get_object(container, obj, query_string='multipart-manifest=get')
        manifest = parse_api_response(headers, manifest)
        for chunk in manifest:
            if chunk.get('sub_slo'):
                scont, sobj = chunk['name'].lstrip('/').split('/', 1)
                chunks.extend(self._get_chunk_data(conn, scont, sobj, {'x-static-large-object': True}))
            else:
                chunks.append(chunk)
    else:
        chunks.append({'hash': headers.get('etag').strip('"'), 'bytes': int(headers.get('content-length'))})
    return chunks