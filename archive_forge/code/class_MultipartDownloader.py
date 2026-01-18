import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
class MultipartDownloader(object):

    def __init__(self, client, config, osutil, executor_cls=concurrent.futures.ThreadPoolExecutor):
        self._client = client
        self._config = config
        self._os = osutil
        self._executor_cls = executor_cls
        self._ioqueue = ShutdownQueue(self._config.max_io_queue)

    def download_file(self, bucket, key, filename, object_size, extra_args, callback=None):
        with self._executor_cls(max_workers=2) as controller:
            download_parts_handler = functools.partial(self._download_file_as_future, bucket, key, filename, object_size, callback)
            parts_future = controller.submit(download_parts_handler)
            io_writes_handler = functools.partial(self._perform_io_writes, filename)
            io_future = controller.submit(io_writes_handler)
            results = concurrent.futures.wait([parts_future, io_future], return_when=concurrent.futures.FIRST_EXCEPTION)
            self._process_future_results(results)

    def _process_future_results(self, futures):
        finished, unfinished = futures
        for future in finished:
            future.result()

    def _download_file_as_future(self, bucket, key, filename, object_size, callback):
        part_size = self._config.multipart_chunksize
        num_parts = int(math.ceil(object_size / float(part_size)))
        max_workers = self._config.max_concurrency
        download_partial = functools.partial(self._download_range, bucket, key, filename, part_size, num_parts, callback)
        try:
            with self._executor_cls(max_workers=max_workers) as executor:
                list(executor.map(download_partial, range(num_parts)))
        finally:
            self._ioqueue.put(SHUTDOWN_SENTINEL)

    def _calculate_range_param(self, part_size, part_index, num_parts):
        start_range = part_index * part_size
        if part_index == num_parts - 1:
            end_range = ''
        else:
            end_range = start_range + part_size - 1
        range_param = 'bytes=%s-%s' % (start_range, end_range)
        return range_param

    def _download_range(self, bucket, key, filename, part_size, num_parts, callback, part_index):
        try:
            range_param = self._calculate_range_param(part_size, part_index, num_parts)
            max_attempts = self._config.num_download_attempts
            last_exception = None
            for i in range(max_attempts):
                try:
                    logger.debug('Making get_object call.')
                    response = self._client.get_object(Bucket=bucket, Key=key, Range=range_param)
                    streaming_body = StreamReaderProgress(response['Body'], callback)
                    buffer_size = 1024 * 16
                    current_index = part_size * part_index
                    for chunk in iter(lambda: streaming_body.read(buffer_size), b''):
                        self._ioqueue.put((current_index, chunk))
                        current_index += len(chunk)
                    return
                except (socket.timeout, socket.error, ReadTimeoutError, IncompleteReadError) as e:
                    logger.debug('Retrying exception caught (%s), retrying request, (attempt %s / %s)', e, i, max_attempts, exc_info=True)
                    last_exception = e
                    continue
            raise RetriesExceededError(last_exception)
        finally:
            logger.debug('EXITING _download_range for part: %s', part_index)

    def _perform_io_writes(self, filename):
        with self._os.open(filename, 'wb') as f:
            while True:
                task = self._ioqueue.get()
                if task is SHUTDOWN_SENTINEL:
                    logger.debug('Shutdown sentinel received in IO handler, shutting down IO handler.')
                    return
                else:
                    try:
                        offset, data = task
                        f.seek(offset)
                        f.write(data)
                    except Exception as e:
                        logger.debug('Caught exception in IO thread: %s', e, exc_info=True)
                        self._ioqueue.trigger_shutdown()
                        raise