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
def _wrap_data(self, data, callbacks, close_callbacks):
    """
        Wraps data with the interrupt reader and the file chunk reader.

        :type data: bytes
        :param data: The data to wrap.

        :type callbacks: list
        :param callbacks: The callbacks associated with the transfer future.

        :type close_callbacks: list
        :param close_callbacks: The callbacks to be called when closing the
            wrapper for the data.

        :return: Fully wrapped data.
        """
    fileobj = self._wrap_fileobj(six.BytesIO(data))
    return self._osutil.open_file_chunk_reader_from_fileobj(fileobj=fileobj, chunk_size=len(data), full_file_size=len(data), callbacks=callbacks, close_callbacks=close_callbacks)