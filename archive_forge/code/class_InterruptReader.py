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
class InterruptReader(object):
    """Wrapper that can interrupt reading using an error

    It uses a transfer coordinator to propagate an error if it notices
    that a read is being made while the file is being read from.

    :type fileobj: file-like obj
    :param fileobj: The file-like object to read from

    :type transfer_coordinator: s3transfer.futures.TransferCoordinator
    :param transfer_coordinator: The transfer coordinator to use if the
        reader needs to be interrupted.
    """

    def __init__(self, fileobj, transfer_coordinator):
        self._fileobj = fileobj
        self._transfer_coordinator = transfer_coordinator

    def read(self, amount=None):
        if self._transfer_coordinator.exception:
            raise self._transfer_coordinator.exception
        return self._fileobj.read(amount)

    def seek(self, where):
        self._fileobj.seek(where)

    def tell(self):
        return self._fileobj.tell()

    def close(self):
        self._fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()