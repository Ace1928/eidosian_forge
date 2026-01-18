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
def _submit_upload_request(self, client, config, osutil, request_executor, transfer_future, upload_input_manager):
    call_args = transfer_future.meta.call_args
    put_object_tag = self._get_upload_task_tag(upload_input_manager, 'put_object')
    self._transfer_coordinator.submit(request_executor, PutObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'fileobj': upload_input_manager.get_put_object_body(transfer_future), 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': call_args.extra_args}, is_final=True), tag=put_object_tag)