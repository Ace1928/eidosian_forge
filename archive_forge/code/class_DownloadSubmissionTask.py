import logging
import threading
import heapq
from botocore.compat import six
from s3transfer.compat import seekable
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import get_callbacks
from s3transfer.utils import invoke_progress_callbacks
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import FunctionContainer
from s3transfer.utils import CountCallbackInvoker
from s3transfer.utils import StreamReaderProgress
from s3transfer.utils import DeferredOpenFile
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
class DownloadSubmissionTask(SubmissionTask):
    """Task for submitting tasks to execute a download"""

    def _get_download_output_manager_cls(self, transfer_future, osutil):
        """Retrieves a class for managing output for a download

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future for the request

        :type osutil: s3transfer.utils.OSUtils
        :param osutil: The os utility associated to the transfer

        :rtype: class of DownloadOutputManager
        :returns: The appropriate class to use for managing a specific type of
            input for downloads.
        """
        download_manager_resolver_chain = [DownloadSpecialFilenameOutputManager, DownloadFilenameOutputManager, DownloadSeekableOutputManager, DownloadNonSeekableOutputManager]
        fileobj = transfer_future.meta.call_args.fileobj
        for download_manager_cls in download_manager_resolver_chain:
            if download_manager_cls.is_compatible(fileobj, osutil):
                return download_manager_cls
        raise RuntimeError('Output %s of type: %s is not supported.' % (fileobj, type(fileobj)))

    def _submit(self, client, config, osutil, request_executor, io_executor, transfer_future, bandwidth_limiter=None):
        """
        :param client: The client associated with the transfer manager

        :type config: s3transfer.manager.TransferConfig
        :param config: The transfer config associated with the transfer
            manager

        :type osutil: s3transfer.utils.OSUtil
        :param osutil: The os utility associated to the transfer manager

        :type request_executor: s3transfer.futures.BoundedExecutor
        :param request_executor: The request executor associated with the
            transfer manager

        :type io_executor: s3transfer.futures.BoundedExecutor
        :param io_executor: The io executor associated with the
            transfer manager

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future associated with the
            transfer request that tasks are being submitted for

        :type bandwidth_limiter: s3transfer.bandwidth.BandwidthLimiter
        :param bandwidth_limiter: The bandwidth limiter to use when
            downloading streams
        """
        if transfer_future.meta.size is None:
            response = client.head_object(Bucket=transfer_future.meta.call_args.bucket, Key=transfer_future.meta.call_args.key, **transfer_future.meta.call_args.extra_args)
            transfer_future.meta.provide_transfer_size(response['ContentLength'])
        download_output_manager = self._get_download_output_manager_cls(transfer_future, osutil)(osutil, self._transfer_coordinator, io_executor)
        if transfer_future.meta.size < config.multipart_threshold:
            self._submit_download_request(client, config, osutil, request_executor, io_executor, download_output_manager, transfer_future, bandwidth_limiter)
        else:
            self._submit_ranged_download_request(client, config, osutil, request_executor, io_executor, download_output_manager, transfer_future, bandwidth_limiter)

    def _submit_download_request(self, client, config, osutil, request_executor, io_executor, download_output_manager, transfer_future, bandwidth_limiter):
        call_args = transfer_future.meta.call_args
        fileobj = download_output_manager.get_fileobj_for_io_writes(transfer_future)
        progress_callbacks = get_callbacks(transfer_future, 'progress')
        get_object_tag = download_output_manager.get_download_task_tag()
        final_task = download_output_manager.get_final_io_task()
        self._transfer_coordinator.submit(request_executor, ImmediatelyWriteIOGetObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'fileobj': fileobj, 'extra_args': call_args.extra_args, 'callbacks': progress_callbacks, 'max_attempts': config.num_download_attempts, 'download_output_manager': download_output_manager, 'io_chunksize': config.io_chunksize, 'bandwidth_limiter': bandwidth_limiter}, done_callbacks=[final_task]), tag=get_object_tag)

    def _submit_ranged_download_request(self, client, config, osutil, request_executor, io_executor, download_output_manager, transfer_future, bandwidth_limiter):
        call_args = transfer_future.meta.call_args
        progress_callbacks = get_callbacks(transfer_future, 'progress')
        fileobj = download_output_manager.get_fileobj_for_io_writes(transfer_future)
        part_size = config.multipart_chunksize
        num_parts = calculate_num_parts(transfer_future.meta.size, part_size)
        get_object_tag = download_output_manager.get_download_task_tag()
        finalize_download_invoker = CountCallbackInvoker(self._get_final_io_task_submission_callback(download_output_manager, io_executor))
        for i in range(num_parts):
            range_parameter = calculate_range_parameter(part_size, i, num_parts)
            extra_args = {'Range': range_parameter}
            extra_args.update(call_args.extra_args)
            finalize_download_invoker.increment()
            self._transfer_coordinator.submit(request_executor, GetObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'fileobj': fileobj, 'extra_args': extra_args, 'callbacks': progress_callbacks, 'max_attempts': config.num_download_attempts, 'start_index': i * part_size, 'download_output_manager': download_output_manager, 'io_chunksize': config.io_chunksize, 'bandwidth_limiter': bandwidth_limiter}, done_callbacks=[finalize_download_invoker.decrement]), tag=get_object_tag)
        finalize_download_invoker.finalize()

    def _get_final_io_task_submission_callback(self, download_manager, io_executor):
        final_task = download_manager.get_final_io_task()
        return FunctionContainer(self._transfer_coordinator.submit, io_executor, final_task)

    def _calculate_range_param(self, part_size, part_index, num_parts):
        start_range = part_index * part_size
        if part_index == num_parts - 1:
            end_range = ''
        else:
            end_range = start_range + part_size - 1
        range_param = 'bytes=%s-%s' % (start_range, end_range)
        return range_param