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
def _upload_stream_segment(conn, container, object_name, segment_container, segment_name, segment_size, segment_index, headers, fd):
    """
        Upload a segment from a stream, buffering it in memory first. The
        resulting object is placed either as a segment in the segment
        container, or if it is smaller than a single segment, as the given
        object name.

        :param conn: Swift Connection to use.
        :param container: Container in which the object would be placed.
        :param object_name: Name of the final object (used in case the stream
                            is smaller than the segment_size)
        :param segment_container: Container to hold the object segments.
        :param segment_name: The name of the segment.
        :param segment_size: Minimum segment size.
        :param segment_index: The segment index.
        :param headers: Headers to attach to the segment/object.
        :param fd: File-like handle for the content. Must implement read().

        :returns: Dictionary, containing the following keys:
                    - complete -- whether the stream is exhausted
                    - segment_size - the actual size of the segment (may be
                                     smaller than the passed in segment_size)
                    - segment_location - path to the segment
                    - segment_index - index of the segment
                    - segment_etag - the ETag for the segment
        """
    buf = []
    dgst = md5()
    bytes_read = 0
    while bytes_read < segment_size:
        data = fd.read(segment_size - bytes_read)
        if not data:
            break
        bytes_read += len(data)
        dgst.update(data)
        buf.append(data)
    buf = b''.join(buf)
    segment_hash = dgst.hexdigest()
    if not buf and segment_index > 0:
        return {'complete': True, 'segment_size': 0, 'segment_index': None, 'segment_etag': None, 'segment_location': None, 'success': True}
    if segment_index == 0 and len(buf) < segment_size:
        ret = SwiftService._put_object(conn, container, object_name, buf, headers, segment_hash)
        ret['segment_location'] = '/%s/%s' % (container, object_name)
    else:
        ret = SwiftService._put_object(conn, segment_container, segment_name, buf, headers, segment_hash)
        ret['segment_location'] = '/%s/%s' % (segment_container, segment_name)
    ret.update(dict(complete=len(buf) < segment_size, segment_size=len(buf), segment_index=segment_index, segment_etag=segment_hash, for_object=object_name))
    return ret