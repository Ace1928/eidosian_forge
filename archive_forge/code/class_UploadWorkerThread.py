import os
import math
import threading
import hashlib
import time
import logging
from boto.compat import Queue
import binascii
from boto.glacier.utils import DEFAULT_PART_SIZE, minimum_part_size, \
from boto.glacier.exceptions import UploadArchiveError, \
class UploadWorkerThread(TransferThread):

    def __init__(self, api, vault_name, filename, upload_id, worker_queue, result_queue, num_retries=5, time_between_retries=5, retry_exceptions=Exception):
        super(UploadWorkerThread, self).__init__(worker_queue, result_queue)
        self._api = api
        self._vault_name = vault_name
        self._filename = filename
        self._fileobj = open(filename, 'rb')
        self._upload_id = upload_id
        self._num_retries = num_retries
        self._time_between_retries = time_between_retries
        self._retry_exceptions = retry_exceptions

    def _process_chunk(self, work):
        result = None
        for i in range(self._num_retries + 1):
            try:
                result = self._upload_chunk(work)
                break
            except self._retry_exceptions as e:
                log.error('Exception caught uploading part number %s for vault %s, attempt: (%s / %s), filename: %s, exception: %s, msg: %s', work[0], self._vault_name, i + 1, self._num_retries + 1, self._filename, e.__class__, e)
                time.sleep(self._time_between_retries)
                result = e
        return result

    def _upload_chunk(self, work):
        part_number, part_size = work
        start_byte = part_number * part_size
        self._fileobj.seek(start_byte)
        contents = self._fileobj.read(part_size)
        linear_hash = hashlib.sha256(contents).hexdigest()
        tree_hash_bytes = tree_hash(chunk_hashes(contents))
        byte_range = (start_byte, start_byte + len(contents) - 1)
        log.debug('Uploading chunk %s of size %s', part_number, part_size)
        response = self._api.upload_part(self._vault_name, self._upload_id, linear_hash, bytes_to_hex(tree_hash_bytes), byte_range, contents)
        response.read()
        return (part_number, tree_hash_bytes)

    def _cleanup(self):
        self._fileobj.close()