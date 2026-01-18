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
class GetObjectSubmitter(BaseS3TransferProcess):

    def __init__(self, transfer_config, client_factory, transfer_monitor, osutil, download_request_queue, worker_queue):
        """Submit GetObjectJobs to fulfill a download file request

        :param transfer_config: Configuration for transfers.
        :param client_factory: ClientFactory for creating S3 clients.
        :param transfer_monitor: Monitor for notifying and retrieving state
            of transfer.
        :param osutil: OSUtils object to use for os-related behavior when
            performing the transfer.
        :param download_request_queue: Queue to retrieve download file
            requests.
        :param worker_queue: Queue to submit GetObjectJobs for workers
            to perform.
        """
        super(GetObjectSubmitter, self).__init__(client_factory)
        self._transfer_config = transfer_config
        self._transfer_monitor = transfer_monitor
        self._osutil = osutil
        self._download_request_queue = download_request_queue
        self._worker_queue = worker_queue

    def _do_run(self):
        while True:
            download_file_request = self._download_request_queue.get()
            if download_file_request == SHUTDOWN_SIGNAL:
                logger.debug('Submitter shutdown signal received.')
                return
            try:
                self._submit_get_object_jobs(download_file_request)
            except Exception as e:
                logger.debug('Exception caught when submitting jobs for download file request %s: %s', download_file_request, e, exc_info=True)
                self._transfer_monitor.notify_exception(download_file_request.transfer_id, e)
                self._transfer_monitor.notify_done(download_file_request.transfer_id)

    def _submit_get_object_jobs(self, download_file_request):
        size = self._get_size(download_file_request)
        temp_filename = self._allocate_temp_file(download_file_request, size)
        if size < self._transfer_config.multipart_threshold:
            self._submit_single_get_object_job(download_file_request, temp_filename)
        else:
            self._submit_ranged_get_object_jobs(download_file_request, temp_filename, size)

    def _get_size(self, download_file_request):
        expected_size = download_file_request.expected_size
        if expected_size is None:
            expected_size = self._client.head_object(Bucket=download_file_request.bucket, Key=download_file_request.key, **download_file_request.extra_args)['ContentLength']
        return expected_size

    def _allocate_temp_file(self, download_file_request, size):
        temp_filename = self._osutil.get_temp_filename(download_file_request.filename)
        self._osutil.allocate(temp_filename, size)
        return temp_filename

    def _submit_single_get_object_job(self, download_file_request, temp_filename):
        self._notify_jobs_to_complete(download_file_request.transfer_id, 1)
        self._submit_get_object_job(transfer_id=download_file_request.transfer_id, bucket=download_file_request.bucket, key=download_file_request.key, temp_filename=temp_filename, offset=0, extra_args=download_file_request.extra_args, filename=download_file_request.filename)

    def _submit_ranged_get_object_jobs(self, download_file_request, temp_filename, size):
        part_size = self._transfer_config.multipart_chunksize
        num_parts = calculate_num_parts(size, part_size)
        self._notify_jobs_to_complete(download_file_request.transfer_id, num_parts)
        for i in range(num_parts):
            offset = i * part_size
            range_parameter = calculate_range_parameter(part_size, i, num_parts)
            get_object_kwargs = {'Range': range_parameter}
            get_object_kwargs.update(download_file_request.extra_args)
            self._submit_get_object_job(transfer_id=download_file_request.transfer_id, bucket=download_file_request.bucket, key=download_file_request.key, temp_filename=temp_filename, offset=offset, extra_args=get_object_kwargs, filename=download_file_request.filename)

    def _submit_get_object_job(self, **get_object_job_kwargs):
        self._worker_queue.put(GetObjectJob(**get_object_job_kwargs))

    def _notify_jobs_to_complete(self, transfer_id, jobs_to_complete):
        logger.debug('Notifying %s job(s) to complete for transfer_id %s.', jobs_to_complete, transfer_id)
        self._transfer_monitor.notify_expected_jobs_to_complete(transfer_id, jobs_to_complete)