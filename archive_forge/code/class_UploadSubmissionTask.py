import math
from botocore.compat import six
from s3transfer.compat import seekable, readable
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import DeferredOpenFile, ChunksizeAdjuster
class UploadSubmissionTask(SubmissionTask):
    """Task for submitting tasks to execute an upload"""
    UPLOAD_PART_ARGS = ['SSECustomerKey', 'SSECustomerAlgorithm', 'SSECustomerKeyMD5', 'RequestPayer']
    COMPLETE_MULTIPART_ARGS = ['RequestPayer']

    def _get_upload_input_manager_cls(self, transfer_future):
        """Retrieves a class for managing input for an upload based on file type

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future for the request

        :rtype: class of UploadInputManager
        :returns: The appropriate class to use for managing a specific type of
            input for uploads.
        """
        upload_manager_resolver_chain = [UploadFilenameInputManager, UploadSeekableInputManager, UploadNonSeekableInputManager]
        fileobj = transfer_future.meta.call_args.fileobj
        for upload_manager_cls in upload_manager_resolver_chain:
            if upload_manager_cls.is_compatible(fileobj):
                return upload_manager_cls
        raise RuntimeError('Input %s of type: %s is not supported.' % (fileobj, type(fileobj)))

    def _submit(self, client, config, osutil, request_executor, transfer_future, bandwidth_limiter=None):
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

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future associated with the
            transfer request that tasks are being submitted for
        """
        upload_input_manager = self._get_upload_input_manager_cls(transfer_future)(osutil, self._transfer_coordinator, bandwidth_limiter)
        if transfer_future.meta.size is None:
            upload_input_manager.provide_transfer_size(transfer_future)
        if not upload_input_manager.requires_multipart_upload(transfer_future, config):
            self._submit_upload_request(client, config, osutil, request_executor, transfer_future, upload_input_manager)
        else:
            self._submit_multipart_request(client, config, osutil, request_executor, transfer_future, upload_input_manager)

    def _submit_upload_request(self, client, config, osutil, request_executor, transfer_future, upload_input_manager):
        call_args = transfer_future.meta.call_args
        put_object_tag = self._get_upload_task_tag(upload_input_manager, 'put_object')
        self._transfer_coordinator.submit(request_executor, PutObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'fileobj': upload_input_manager.get_put_object_body(transfer_future), 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': call_args.extra_args}, is_final=True), tag=put_object_tag)

    def _submit_multipart_request(self, client, config, osutil, request_executor, transfer_future, upload_input_manager):
        call_args = transfer_future.meta.call_args
        create_multipart_future = self._transfer_coordinator.submit(request_executor, CreateMultipartUploadTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': call_args.extra_args}))
        part_futures = []
        extra_part_args = self._extra_upload_part_args(call_args.extra_args)
        upload_part_tag = self._get_upload_task_tag(upload_input_manager, 'upload_part')
        size = transfer_future.meta.size
        adjuster = ChunksizeAdjuster()
        chunksize = adjuster.adjust_chunksize(config.multipart_chunksize, size)
        part_iterator = upload_input_manager.yield_upload_part_bodies(transfer_future, chunksize)
        for part_number, fileobj in part_iterator:
            part_futures.append(self._transfer_coordinator.submit(request_executor, UploadPartTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'fileobj': fileobj, 'bucket': call_args.bucket, 'key': call_args.key, 'part_number': part_number, 'extra_args': extra_part_args}, pending_main_kwargs={'upload_id': create_multipart_future}), tag=upload_part_tag))
        complete_multipart_extra_args = self._extra_complete_multipart_args(call_args.extra_args)
        self._transfer_coordinator.submit(request_executor, CompleteMultipartUploadTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': complete_multipart_extra_args}, pending_main_kwargs={'upload_id': create_multipart_future, 'parts': part_futures}, is_final=True))

    def _extra_upload_part_args(self, extra_args):
        return get_filtered_dict(extra_args, self.UPLOAD_PART_ARGS)

    def _extra_complete_multipart_args(self, extra_args):
        return get_filtered_dict(extra_args, self.COMPLETE_MULTIPART_ARGS)

    def _get_upload_task_tag(self, upload_input_manager, operation_name):
        tag = None
        if upload_input_manager.stores_body_in_memory(operation_name):
            tag = IN_MEMORY_UPLOAD_TAG
        return tag