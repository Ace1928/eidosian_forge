import copy
import math
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import ChunksizeAdjuster
class CopySubmissionTask(SubmissionTask):
    """Task for submitting tasks to execute a copy"""
    EXTRA_ARGS_TO_HEAD_ARGS_MAPPING = {'CopySourceIfMatch': 'IfMatch', 'CopySourceIfModifiedSince': 'IfModifiedSince', 'CopySourceIfNoneMatch': 'IfNoneMatch', 'CopySourceIfUnmodifiedSince': 'IfUnmodifiedSince', 'CopySourceSSECustomerKey': 'SSECustomerKey', 'CopySourceSSECustomerAlgorithm': 'SSECustomerAlgorithm', 'CopySourceSSECustomerKeyMD5': 'SSECustomerKeyMD5', 'RequestPayer': 'RequestPayer'}
    UPLOAD_PART_COPY_ARGS = ['CopySourceIfMatch', 'CopySourceIfModifiedSince', 'CopySourceIfNoneMatch', 'CopySourceIfUnmodifiedSince', 'CopySourceSSECustomerKey', 'CopySourceSSECustomerAlgorithm', 'CopySourceSSECustomerKeyMD5', 'SSECustomerKey', 'SSECustomerAlgorithm', 'SSECustomerKeyMD5', 'RequestPayer']
    CREATE_MULTIPART_ARGS_BLACKLIST = ['CopySourceIfMatch', 'CopySourceIfModifiedSince', 'CopySourceIfNoneMatch', 'CopySourceIfUnmodifiedSince', 'CopySourceSSECustomerKey', 'CopySourceSSECustomerAlgorithm', 'CopySourceSSECustomerKeyMD5', 'MetadataDirective', 'TaggingDirective']
    COMPLETE_MULTIPART_ARGS = ['RequestPayer']

    def _submit(self, client, config, osutil, request_executor, transfer_future):
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
        if transfer_future.meta.size is None:
            call_args = transfer_future.meta.call_args
            head_object_request = self._get_head_object_request_from_copy_source(call_args.copy_source)
            extra_args = call_args.extra_args
            for param, value in extra_args.items():
                if param in self.EXTRA_ARGS_TO_HEAD_ARGS_MAPPING:
                    head_object_request[self.EXTRA_ARGS_TO_HEAD_ARGS_MAPPING[param]] = value
            response = call_args.source_client.head_object(**head_object_request)
            transfer_future.meta.provide_transfer_size(response['ContentLength'])
        if transfer_future.meta.size < config.multipart_threshold:
            self._submit_copy_request(client, config, osutil, request_executor, transfer_future)
        else:
            self._submit_multipart_request(client, config, osutil, request_executor, transfer_future)

    def _submit_copy_request(self, client, config, osutil, request_executor, transfer_future):
        call_args = transfer_future.meta.call_args
        progress_callbacks = get_callbacks(transfer_future, 'progress')
        self._transfer_coordinator.submit(request_executor, CopyObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'copy_source': call_args.copy_source, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': call_args.extra_args, 'callbacks': progress_callbacks, 'size': transfer_future.meta.size}, is_final=True))

    def _submit_multipart_request(self, client, config, osutil, request_executor, transfer_future):
        call_args = transfer_future.meta.call_args
        create_multipart_extra_args = {}
        for param, val in call_args.extra_args.items():
            if param not in self.CREATE_MULTIPART_ARGS_BLACKLIST:
                create_multipart_extra_args[param] = val
        create_multipart_future = self._transfer_coordinator.submit(request_executor, CreateMultipartUploadTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': create_multipart_extra_args}))
        part_size = config.multipart_chunksize
        adjuster = ChunksizeAdjuster()
        part_size = adjuster.adjust_chunksize(part_size, transfer_future.meta.size)
        num_parts = int(math.ceil(transfer_future.meta.size / float(part_size)))
        part_futures = []
        progress_callbacks = get_callbacks(transfer_future, 'progress')
        for part_number in range(1, num_parts + 1):
            extra_part_args = self._extra_upload_part_args(call_args.extra_args)
            extra_part_args['CopySourceRange'] = calculate_range_parameter(part_size, part_number - 1, num_parts, transfer_future.meta.size)
            size = self._get_transfer_size(part_size, part_number - 1, num_parts, transfer_future.meta.size)
            part_futures.append(self._transfer_coordinator.submit(request_executor, CopyPartTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'copy_source': call_args.copy_source, 'bucket': call_args.bucket, 'key': call_args.key, 'part_number': part_number, 'extra_args': extra_part_args, 'callbacks': progress_callbacks, 'size': size}, pending_main_kwargs={'upload_id': create_multipart_future})))
        complete_multipart_extra_args = self._extra_complete_multipart_args(call_args.extra_args)
        self._transfer_coordinator.submit(request_executor, CompleteMultipartUploadTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': complete_multipart_extra_args}, pending_main_kwargs={'upload_id': create_multipart_future, 'parts': part_futures}, is_final=True))

    def _get_head_object_request_from_copy_source(self, copy_source):
        if isinstance(copy_source, dict):
            return copy.copy(copy_source)
        else:
            raise TypeError('Expecting dictionary formatted: {"Bucket": bucket_name, "Key": key} but got %s or type %s.' % (copy_source, type(copy_source)))

    def _extra_upload_part_args(self, extra_args):
        return get_filtered_dict(extra_args, self.UPLOAD_PART_COPY_ARGS)

    def _extra_complete_multipart_args(self, extra_args):
        return get_filtered_dict(extra_args, self.COMPLETE_MULTIPART_ARGS)

    def _get_transfer_size(self, part_size, part_index, num_parts, total_transfer_size):
        if part_index == num_parts - 1:
            return total_transfer_size - part_index * part_size
        return part_size