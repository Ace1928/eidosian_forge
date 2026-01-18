from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def _get_file_internal(self, fp, headers=None, cb=None, num_cb=10, torrent=False, version_id=None, override_num_retries=None, response_headers=None, hash_algs=None, query_args=None):
    if headers is None:
        headers = {}
    save_debug = self.bucket.connection.debug
    if self.bucket.connection.debug == 1:
        self.bucket.connection.debug = 0
    query_args = query_args or []
    if torrent:
        query_args.append('torrent')
    if hash_algs is None and (not torrent):
        hash_algs = {'md5': md5}
    digesters = dict(((alg, hash_algs[alg]()) for alg in hash_algs or {}))
    if version_id is None:
        version_id = self.version_id
    if version_id:
        query_args.append('versionId=%s' % version_id)
    if response_headers:
        for key in response_headers:
            query_args.append('%s=%s' % (key, urllib.parse.quote(response_headers[key])))
    query_args = '&'.join(query_args)
    self.open('r', headers, query_args=query_args, override_num_retries=override_num_retries)
    data_len = 0
    if cb:
        if self.size is None:
            cb_size = 0
        else:
            cb_size = self.size
        if self.size is None and num_cb != -1:
            cb_count = 1024 * 1024 / self.BufferSize
        elif num_cb > 1:
            cb_count = int(math.ceil(cb_size / self.BufferSize / (num_cb - 1.0)))
        elif num_cb < 0:
            cb_count = -1
        else:
            cb_count = 0
        i = 0
        cb(data_len, cb_size)
    try:
        for key_bytes in self:
            print_to_fd(six.ensure_binary(key_bytes), file=fp, end=b'')
            data_len += len(key_bytes)
            for alg in digesters:
                digesters[alg].update(key_bytes)
            if cb:
                if cb_size > 0 and data_len >= cb_size:
                    break
                i += 1
                if i == cb_count or cb_count == -1:
                    cb(data_len, cb_size)
                    i = 0
        if hasattr(self, '_size_of_range') and data_len < self._size_of_range:
            raise ResumableDownloadException('Download stream truncated. Received {} of {} bytes.'.format(data_len, self._size_of_range), ResumableTransferDisposition.WAIT_BEFORE_RETRY)
    except IOError as e:
        if e.errno == errno.ENOSPC:
            raise StorageDataError('Out of space for destination file %s' % fp.name)
        raise
    if cb and (cb_count <= 1 or i > 0) and (data_len > 0):
        cb(data_len, cb_size)
    for alg in digesters:
        self.local_hashes[alg] = digesters[alg].digest()
    if self.size is None and (not torrent) and ('Range' not in headers):
        self.size = data_len
    self.close()
    self.bucket.connection.debug = save_debug