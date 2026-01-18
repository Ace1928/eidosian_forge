import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
class ChunkStreamReader:

    def __init__(self, file_path, start_block, end_block, chunk_size):
        self.fd = open(file_path, 'rb')
        self.fd.seek(start_block)
        self.start_block = start_block
        self.end_block = end_block
        self.chunk_size = chunk_size
        self.bytes_read = 0
        self.stop_iteration = False

        def close_file(fd):
            try:
                fd.close()
            except Exception:
                pass
        atexit.register(close_file, self.fd)

    def __iter__(self):
        return self

    def next(self):
        if self.stop_iteration:
            self.fd.close()
            raise StopIteration
        block_size = self.chunk_size
        if self.bytes_read + block_size > self.end_block - self.start_block:
            block_size = self.end_block - self.start_block - self.bytes_read
            self.stop_iteration = True
        block = self.fd.read(block_size)
        self.bytes_read += block_size
        return block

    def __next__(self):
        return self.next()