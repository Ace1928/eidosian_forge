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
class UploadFilenameInputManager(UploadInputManager):
    """Upload utility for filenames"""

    @classmethod
    def is_compatible(cls, upload_source):
        return isinstance(upload_source, six.string_types)

    def stores_body_in_memory(self, operation_name):
        return False

    def provide_transfer_size(self, transfer_future):
        transfer_future.meta.provide_transfer_size(self._osutil.get_file_size(transfer_future.meta.call_args.fileobj))

    def requires_multipart_upload(self, transfer_future, config):
        return transfer_future.meta.size >= config.multipart_threshold

    def get_put_object_body(self, transfer_future):
        fileobj, full_size = self._get_put_object_fileobj_with_full_size(transfer_future)
        fileobj = self._wrap_fileobj(fileobj)
        callbacks = self._get_progress_callbacks(transfer_future)
        close_callbacks = self._get_close_callbacks(callbacks)
        size = transfer_future.meta.size
        return self._osutil.open_file_chunk_reader_from_fileobj(fileobj=fileobj, chunk_size=size, full_file_size=full_size, callbacks=callbacks, close_callbacks=close_callbacks)

    def yield_upload_part_bodies(self, transfer_future, chunksize):
        full_file_size = transfer_future.meta.size
        num_parts = self._get_num_parts(transfer_future, chunksize)
        for part_number in range(1, num_parts + 1):
            callbacks = self._get_progress_callbacks(transfer_future)
            close_callbacks = self._get_close_callbacks(callbacks)
            start_byte = chunksize * (part_number - 1)
            fileobj, full_size = self._get_upload_part_fileobj_with_full_size(transfer_future.meta.call_args.fileobj, start_byte=start_byte, part_size=chunksize, full_file_size=full_file_size)
            fileobj = self._wrap_fileobj(fileobj)
            read_file_chunk = self._osutil.open_file_chunk_reader_from_fileobj(fileobj=fileobj, chunk_size=chunksize, full_file_size=full_size, callbacks=callbacks, close_callbacks=close_callbacks)
            yield (part_number, read_file_chunk)

    def _get_deferred_open_file(self, fileobj, start_byte):
        fileobj = DeferredOpenFile(fileobj, start_byte, open_function=self._osutil.open)
        return fileobj

    def _get_put_object_fileobj_with_full_size(self, transfer_future):
        fileobj = transfer_future.meta.call_args.fileobj
        size = transfer_future.meta.size
        return (self._get_deferred_open_file(fileobj, 0), size)

    def _get_upload_part_fileobj_with_full_size(self, fileobj, **kwargs):
        start_byte = kwargs['start_byte']
        full_size = kwargs['full_file_size']
        return (self._get_deferred_open_file(fileobj, start_byte), full_size)

    def _get_num_parts(self, transfer_future, part_size):
        return int(math.ceil(transfer_future.meta.size / float(part_size)))