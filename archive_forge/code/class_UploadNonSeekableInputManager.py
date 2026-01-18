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
class UploadNonSeekableInputManager(UploadInputManager):
    """Upload utility for a file-like object that cannot seek."""

    def __init__(self, osutil, transfer_coordinator, bandwidth_limiter=None):
        super(UploadNonSeekableInputManager, self).__init__(osutil, transfer_coordinator, bandwidth_limiter)
        self._initial_data = b''

    @classmethod
    def is_compatible(cls, upload_source):
        return readable(upload_source)

    def stores_body_in_memory(self, operation_name):
        return True

    def provide_transfer_size(self, transfer_future):
        return

    def requires_multipart_upload(self, transfer_future, config):
        if transfer_future.meta.size is not None:
            return transfer_future.meta.size >= config.multipart_threshold
        fileobj = transfer_future.meta.call_args.fileobj
        threshold = config.multipart_threshold
        self._initial_data = self._read(fileobj, threshold, False)
        if len(self._initial_data) < threshold:
            return False
        else:
            return True

    def get_put_object_body(self, transfer_future):
        callbacks = self._get_progress_callbacks(transfer_future)
        close_callbacks = self._get_close_callbacks(callbacks)
        fileobj = transfer_future.meta.call_args.fileobj
        body = self._wrap_data(self._initial_data + fileobj.read(), callbacks, close_callbacks)
        self._initial_data = None
        return body

    def yield_upload_part_bodies(self, transfer_future, chunksize):
        file_object = transfer_future.meta.call_args.fileobj
        part_number = 0
        while True:
            callbacks = self._get_progress_callbacks(transfer_future)
            close_callbacks = self._get_close_callbacks(callbacks)
            part_number += 1
            part_content = self._read(file_object, chunksize)
            if not part_content:
                break
            part_object = self._wrap_data(part_content, callbacks, close_callbacks)
            part_content = None
            yield (part_number, part_object)

    def _read(self, fileobj, amount, truncate=True):
        """
        Reads a specific amount of data from a stream and returns it. If there
        is any data in initial_data, that will be popped out first.

        :type fileobj: A file-like object that implements read
        :param fileobj: The stream to read from.

        :type amount: int
        :param amount: The number of bytes to read from the stream.

        :type truncate: bool
        :param truncate: Whether or not to truncate initial_data after
            reading from it.

        :return: Generator which generates part bodies from the initial data.
        """
        if len(self._initial_data) == 0:
            return fileobj.read(amount)
        if amount <= len(self._initial_data):
            data = self._initial_data[:amount]
            if truncate:
                self._initial_data = self._initial_data[amount:]
            return data
        amount_to_read = amount - len(self._initial_data)
        data = self._initial_data + fileobj.read(amount_to_read)
        if truncate:
            self._initial_data = b''
        return data

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