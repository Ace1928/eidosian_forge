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
class ConcurrentDownloader(ConcurrentTransferer):
    """
    Concurrently download an archive from glacier.

    This class uses a thread pool to concurrently download an archive
    from glacier.

    The threadpool is completely managed by this class and is
    transparent to the users of this class.

    """

    def __init__(self, job, part_size=DEFAULT_PART_SIZE, num_threads=10):
        """
        :param job: A layer2 job object for archive retrieval object.

        :param part_size: The size, in bytes, of the chunks to use when uploading
            the archive parts.  The part size must be a megabyte multiplied by
            a power of two.

        """
        super(ConcurrentDownloader, self).__init__(part_size, num_threads)
        self._job = job

    def download(self, filename):
        """
        Concurrently download an archive.

        :param filename: The filename to download the archive to
        :type filename: str

        """
        total_size = self._job.archive_size
        total_parts, part_size = self._calculate_required_part_size(total_size)
        worker_queue = Queue()
        result_queue = Queue()
        self._add_work_items_to_queue(total_parts, worker_queue, part_size)
        self._start_download_threads(result_queue, worker_queue)
        try:
            self._wait_for_download_threads(filename, result_queue, total_parts)
        except DownloadArchiveError as e:
            log.debug('An error occurred while downloading an archive: %s', e)
            raise e
        log.debug('Download completed.')

    def _wait_for_download_threads(self, filename, result_queue, total_parts):
        """
        Waits until the result_queue is filled with all the downloaded parts
        This indicates that all part downloads have completed

        Saves downloaded parts into filename

        :param filename:
        :param result_queue:
        :param total_parts:
        """
        hash_chunks = [None] * total_parts
        with open(filename, 'wb') as f:
            for _ in range(total_parts):
                result = result_queue.get()
                if isinstance(result, Exception):
                    log.debug('An error was found in the result queue, terminating threads: %s', result)
                    self._shutdown_threads()
                    raise DownloadArchiveError('An error occurred while uploading an archive: %s' % result)
                part_number, part_size, actual_hash, data = result
                hash_chunks[part_number] = actual_hash
                start_byte = part_number * part_size
                f.seek(start_byte)
                f.write(data)
                f.flush()
        final_hash = bytes_to_hex(tree_hash(hash_chunks))
        log.debug('Verifying final tree hash of archive, expecting: %s, actual: %s', self._job.sha256_treehash, final_hash)
        if self._job.sha256_treehash != final_hash:
            self._shutdown_threads()
            raise TreeHashDoesNotMatchError('Tree hash for entire archive does not match, expected: %s, got: %s' % (self._job.sha256_treehash, final_hash))
        self._shutdown_threads()

    def _start_download_threads(self, result_queue, worker_queue):
        log.debug('Starting threads.')
        for _ in range(self._num_threads):
            thread = DownloadWorkerThread(self._job, worker_queue, result_queue)
            time.sleep(0.2)
            thread.start()
            self._threads.append(thread)