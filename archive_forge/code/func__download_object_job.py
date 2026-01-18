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
def _download_object_job(self, conn, container, obj, options):
    out_file = options['out_file']
    results_dict = {}
    req_headers = split_headers(options['header'], '')
    pseudodir = False
    path = join(container, obj) if options['yes_all'] else obj
    path = path.lstrip(os_path_sep)
    options['skip_identical'] = options['skip_identical'] and out_file != '-'
    if options['prefix'] and options['remove_prefix']:
        path = path[len(options['prefix']):].lstrip('/')
    if options['out_directory']:
        path = os.path.join(options['out_directory'], path)
    if options['skip_identical']:
        filename = out_file if out_file else path
        try:
            fp = open(filename, 'rb', DISK_BUFFER)
        except IOError:
            pass
        else:
            with fp:
                md5sum = md5()
                while True:
                    data = fp.read(DISK_BUFFER)
                    if not data:
                        break
                    md5sum.update(data)
                req_headers['If-None-Match'] = md5sum.hexdigest()
    try:
        start_time = time()
        get_args = {'resp_chunk_size': DISK_BUFFER, 'headers': req_headers, 'response_dict': results_dict}
        if options.get('version_id') is not None:
            get_args['query_string'] = 'version-id=%s' % options['version_id']
        if options['skip_identical']:
            get_args['query_string'] = 'multipart-manifest=get'
        try:
            headers, body = conn.get_object(container, obj, **get_args)
        except ClientException as e:
            if not options['skip_identical']:
                raise
            if e.http_status != 304:
                raise
            headers = results_dict['headers']
            if 'x-object-manifest' in headers:
                body = []
            elif config_true_value(headers.get('x-static-large-object')):
                body = [b'[]']
            else:
                raise
        if options['skip_identical']:
            if config_true_value(headers.get('x-static-large-object')) or 'x-object-manifest' in headers:
                chunk_data = self._get_chunk_data(conn, container, obj, headers, b''.join(body))
            else:
                chunk_data = None
            if chunk_data is not None:
                if self._is_identical(chunk_data, filename):
                    raise ClientException('Large object is identical', http_status=304)
                del get_args['query_string']
                get_args['response_dict'].clear()
                headers, body = conn.get_object(container, obj, **get_args)
        headers_receipt = time()
        obj_body = _SwiftReader(path, body, headers, options.get('checksum', True))
        no_file = options['no_download']
        if out_file == '-' and (not no_file):
            res = {'action': 'download_object', 'container': container, 'object': obj, 'path': path, 'pseudodir': pseudodir, 'contents': obj_body}
            return res
        fp = None
        try:
            content_type = headers.get('content-type', '').split(';', 1)[0]
            if content_type in KNOWN_DIR_MARKERS:
                make_dir = not no_file and out_file != '-'
                if make_dir and (not isdir(path)):
                    mkdirs(path)
            else:
                make_dir = not (no_file or out_file)
                if make_dir:
                    dirpath = dirname(path)
                    if dirpath and (not isdir(dirpath)):
                        mkdirs(dirpath)
                if not no_file:
                    if out_file:
                        fp = open(out_file, 'wb', DISK_BUFFER)
                    elif basename(path):
                        fp = open(path, 'wb', DISK_BUFFER)
                    else:
                        pseudodir = True
            for chunk in obj_body:
                if fp is not None:
                    fp.write(chunk)
            finish_time = time()
        finally:
            bytes_read = obj_body.bytes_read()
            if fp is not None:
                fp.close()
                if 'x-object-meta-mtime' in headers and (not no_file) and (not options['ignore_mtime']):
                    try:
                        mtime = float(headers['x-object-meta-mtime'])
                    except ValueError:
                        pass
                    else:
                        if options['out_file']:
                            utime(options['out_file'], (mtime, mtime))
                        else:
                            utime(path, (mtime, mtime))
        res = {'action': 'download_object', 'success': True, 'container': container, 'object': obj, 'path': path, 'pseudodir': pseudodir, 'start_time': start_time, 'finish_time': finish_time, 'headers_receipt': headers_receipt, 'auth_end_time': conn.auth_end_time, 'read_length': bytes_read, 'attempts': conn.attempts, 'response_dict': results_dict}
        return res
    except Exception as err:
        traceback, err_time = report_traceback()
        logger.exception(err)
        res = {'action': 'download_object', 'container': container, 'object': obj, 'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': results_dict, 'path': path, 'pseudodir': pseudodir, 'attempts': conn.attempts}
        return res