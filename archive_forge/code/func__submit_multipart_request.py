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