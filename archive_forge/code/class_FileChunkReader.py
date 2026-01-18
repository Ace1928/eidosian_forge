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
class FileChunkReader:

    def __init__(self, file_path, chunk_size):
        self.file_path = file_path
        self.total = os.path.getsize(file_path)
        self.chunk_size = chunk_size
        self.bytes_read = 0
        self.stop_iteration = False

    def __iter__(self):
        return self

    def next(self):
        if self.stop_iteration:
            raise StopIteration
        start_block = self.bytes_read
        end_block = start_block + self.chunk_size
        if end_block >= self.total:
            end_block = self.total
            self.stop_iteration = True
        self.bytes_read += end_block - start_block
        return ChunkStreamReader(file_path=self.file_path, start_block=start_block, end_block=end_block, chunk_size=8192)

    def __next__(self):
        return self.next()