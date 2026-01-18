import collections
import contextlib
import logging
import multiprocessing
import threading
import signal
from copy import deepcopy
import botocore.session
from botocore.config import Config
from s3transfer.constants import MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.constants import PROCESS_USER_AGENT
from s3transfer.compat import MAXINT
from s3transfer.compat import BaseManager
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import BaseTransferFuture
from s3transfer.futures import BaseTransferMeta
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import OSUtils
from s3transfer.utils import CallArgs
class ProcessPoolDownloader(object):

    def __init__(self, client_kwargs=None, config=None):
        """Downloads S3 objects using process pools

        :type client_kwargs: dict
        :param client_kwargs: The keyword arguments to provide when
            instantiating S3 clients. The arguments must match the keyword
            arguments provided to the
            `botocore.session.Session.create_client()` method.

        :type config: ProcessTransferConfig
        :param config: Configuration for the downloader
        """
        if client_kwargs is None:
            client_kwargs = {}
        self._client_factory = ClientFactory(client_kwargs)
        self._transfer_config = config
        if config is None:
            self._transfer_config = ProcessTransferConfig()
        self._download_request_queue = multiprocessing.Queue(1000)
        self._worker_queue = multiprocessing.Queue(1000)
        self._osutil = OSUtils()
        self._started = False
        self._start_lock = threading.Lock()
        self._manager = None
        self._transfer_monitor = None
        self._submitter = None
        self._workers = []

    def download_file(self, bucket, key, filename, extra_args=None, expected_size=None):
        """Downloads the object's contents to a file

        :type bucket: str
        :param bucket: The name of the bucket to download from

        :type key: str
        :param key: The name of the key to download from

        :type filename: str
        :param filename: The name of a file to download to.

        :type extra_args: dict
        :param extra_args: Extra arguments that may be passed to the
            client operation

        :type expected_size: int
        :param expected_size: The expected size in bytes of the download. If
            provided, the downloader will not call HeadObject to determine the
            object's size and use the provided value instead. The size is
            needed to determine whether to do a multipart download.

        :rtype: s3transfer.futures.TransferFuture
        :returns: Transfer future representing the download
        """
        self._start_if_needed()
        if extra_args is None:
            extra_args = {}
        self._validate_all_known_args(extra_args)
        transfer_id = self._transfer_monitor.notify_new_transfer()
        download_file_request = DownloadFileRequest(transfer_id=transfer_id, bucket=bucket, key=key, filename=filename, extra_args=extra_args, expected_size=expected_size)
        logger.debug('Submitting download file request: %s.', download_file_request)
        self._download_request_queue.put(download_file_request)
        call_args = CallArgs(bucket=bucket, key=key, filename=filename, extra_args=extra_args, expected_size=expected_size)
        future = self._get_transfer_future(transfer_id, call_args)
        return future

    def shutdown(self):
        """Shutdown the downloader

        It will wait till all downloads are complete before returning.
        """
        self._shutdown_if_needed()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, *args):
        if isinstance(exc_value, KeyboardInterrupt):
            if self._transfer_monitor is not None:
                self._transfer_monitor.notify_cancel_all_in_progress()
        self.shutdown()

    def _start_if_needed(self):
        with self._start_lock:
            if not self._started:
                self._start()

    def _start(self):
        self._start_transfer_monitor_manager()
        self._start_submitter()
        self._start_get_object_workers()
        self._started = True

    def _validate_all_known_args(self, provided):
        for kwarg in provided:
            if kwarg not in ALLOWED_DOWNLOAD_ARGS:
                raise ValueError("Invalid extra_args key '%s', must be one of: %s" % (kwarg, ', '.join(ALLOWED_DOWNLOAD_ARGS)))

    def _get_transfer_future(self, transfer_id, call_args):
        meta = ProcessPoolTransferMeta(call_args=call_args, transfer_id=transfer_id)
        future = ProcessPoolTransferFuture(monitor=self._transfer_monitor, meta=meta)
        return future

    def _start_transfer_monitor_manager(self):
        logger.debug('Starting the TransferMonitorManager.')
        self._manager = TransferMonitorManager()
        self._manager.start(_add_ignore_handler_for_interrupts)
        self._transfer_monitor = self._manager.TransferMonitor()

    def _start_submitter(self):
        logger.debug('Starting the GetObjectSubmitter.')
        self._submitter = GetObjectSubmitter(transfer_config=self._transfer_config, client_factory=self._client_factory, transfer_monitor=self._transfer_monitor, osutil=self._osutil, download_request_queue=self._download_request_queue, worker_queue=self._worker_queue)
        self._submitter.start()

    def _start_get_object_workers(self):
        logger.debug('Starting %s GetObjectWorkers.', self._transfer_config.max_request_processes)
        for _ in range(self._transfer_config.max_request_processes):
            worker = GetObjectWorker(queue=self._worker_queue, client_factory=self._client_factory, transfer_monitor=self._transfer_monitor, osutil=self._osutil)
            worker.start()
            self._workers.append(worker)

    def _shutdown_if_needed(self):
        with self._start_lock:
            if self._started:
                self._shutdown()

    def _shutdown(self):
        self._shutdown_submitter()
        self._shutdown_get_object_workers()
        self._shutdown_transfer_monitor_manager()
        self._started = False

    def _shutdown_transfer_monitor_manager(self):
        logger.debug('Shutting down the TransferMonitorManager.')
        self._manager.shutdown()

    def _shutdown_submitter(self):
        logger.debug('Shutting down the GetObjectSubmitter.')
        self._download_request_queue.put(SHUTDOWN_SIGNAL)
        self._submitter.join()

    def _shutdown_get_object_workers(self):
        logger.debug('Shutting down the GetObjectWorkers.')
        for _ in self._workers:
            self._worker_queue.put(SHUTDOWN_SIGNAL)
        for worker in self._workers:
            worker.join()