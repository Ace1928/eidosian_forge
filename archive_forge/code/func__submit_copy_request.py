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
def _submit_copy_request(self, client, config, osutil, request_executor, transfer_future):
    call_args = transfer_future.meta.call_args
    progress_callbacks = get_callbacks(transfer_future, 'progress')
    self._transfer_coordinator.submit(request_executor, CopyObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'copy_source': call_args.copy_source, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': call_args.extra_args, 'callbacks': progress_callbacks, 'size': transfer_future.meta.size}, is_final=True))