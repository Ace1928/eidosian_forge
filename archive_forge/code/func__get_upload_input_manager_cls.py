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