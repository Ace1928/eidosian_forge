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
def _wrap_fileobj(self, fileobj):
    fileobj = InterruptReader(fileobj, self._transfer_coordinator)
    if self._bandwidth_limiter:
        fileobj = self._bandwidth_limiter.get_bandwith_limited_stream(fileobj, self._transfer_coordinator, enabled=False)
    return fileobj