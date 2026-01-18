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
class ConcurrentUploader(ConcurrentTransferer):
    """Concurrently upload an archive to glacier.

    This class uses a thread pool to concurrently upload an archive
    to glacier using the multipart upload API.

    The threadpool is completely managed by this class and is
    transparent to the users of this class.

    """

    def __init__(self, api, vault_name, part_size=DEFAULT_PART_SIZE, num_threads=10):
        """
        :type api: :class:`boto.glacier.layer1.Layer1`
        :param api: A layer1 glacier object.

        :type vault_name: str
        :param vault_name: The name of the vault.

        :type part_size: int
        :param part_size: The size, in bytes, of the chunks to use when uploading
            the archive parts.  The part size must be a megabyte multiplied by
            a power of two.

        :type num_threads: int
        :param num_threads: The number of threads to spawn for the thread pool.
            The number of threads will control how much parts are being
            concurrently uploaded.

        """
        super(ConcurrentUploader, self).__init__(part_size, num_threads)
        self._api = api
        self._vault_name = vault_name

    def upload(self, filename, description=None):
        """Concurrently create an archive.

        The part_size value specified when the class was constructed
        will be used *unless* it is smaller than the minimum required
        part size needed for the size of the given file.  In that case,
        the part size used will be the minimum part size required
        to properly upload the given file.

        :type file: str
        :param file: The filename to upload

        :type description: str
        :param description: The description of the archive.

        :rtype: str
        :return: The archive id of the newly created archive.

        """
        total_size = os.stat(filename).st_size
        total_parts, part_size = self._calculate_required_part_size(total_size)
        hash_chunks = [None] * total_parts
        worker_queue = Queue()
        result_queue = Queue()
        response = self._api.initiate_multipart_upload(self._vault_name, part_size, description)
        upload_id = response['UploadId']
        self._add_work_items_to_queue(total_parts, worker_queue, part_size)
        self._start_upload_threads(result_queue, upload_id, worker_queue, filename)
        try:
            self._wait_for_upload_threads(hash_chunks, result_queue, total_parts)
        except UploadArchiveError as e:
            log.debug('An error occurred while uploading an archive, aborting multipart upload.')
            self._api.abort_multipart_upload(self._vault_name, upload_id)
            raise e
        log.debug('Completing upload.')
        response = self._api.complete_multipart_upload(self._vault_name, upload_id, bytes_to_hex(tree_hash(hash_chunks)), total_size)
        log.debug('Upload finished.')
        return response['ArchiveId']

    def _wait_for_upload_threads(self, hash_chunks, result_queue, total_parts):
        for _ in range(total_parts):
            result = result_queue.get()
            if isinstance(result, Exception):
                log.debug('An error was found in the result queue, terminating threads: %s', result)
                self._shutdown_threads()
                raise UploadArchiveError('An error occurred while uploading an archive: %s' % result)
            part_number, tree_sha256 = result
            hash_chunks[part_number] = tree_sha256
        self._shutdown_threads()

    def _start_upload_threads(self, result_queue, upload_id, worker_queue, filename):
        log.debug('Starting threads.')
        for _ in range(self._num_threads):
            thread = UploadWorkerThread(self._api, self._vault_name, filename, upload_id, worker_queue, result_queue)
            time.sleep(0.2)
            thread.start()
            self._threads.append(thread)