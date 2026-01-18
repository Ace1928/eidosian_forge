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
def _upload_slo_manifest(conn, segment_results, container, obj, headers):
    """
        Upload an SLO manifest, given the results of uploading each segment, to
        the specified container.

        :param segment_results: List of response_dict structures, as populated
                                by _upload_segment_job. Specifically, each
                                entry must container the following keys:
                                - segment_location
                                - segment_etag
                                - segment_size
                                - segment_index
        :param container: The container to put the manifest into.
        :param obj: The name of the manifest object to use.
        :param headers: Optional set of headers to attach to the manifest.
        """
    if headers is None:
        headers = {}
    segment_results.sort(key=lambda di: di['segment_index'])
    manifest_data = json.dumps([{'path': d['segment_location'], 'etag': d['segment_etag'], 'size_bytes': d['segment_size']} for d in segment_results])
    response = {}
    conn.put_object(container, obj, manifest_data, headers=headers, query_string='multipart-manifest=put', response_dict=response)
    return response