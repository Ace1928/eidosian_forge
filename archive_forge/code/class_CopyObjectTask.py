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
class CopyObjectTask(Task):
    """Task to do a nonmultipart copy"""

    def _main(self, client, copy_source, bucket, key, extra_args, callbacks, size):
        """
        :param client: The client to use when calling PutObject
        :param copy_source: The CopySource parameter to use
        :param bucket: The name of the bucket to copy to
        :param key: The name of the key to copy to
        :param extra_args: A dictionary of any extra arguments that may be
            used in the upload.
        :param callbacks: List of callbacks to call after copy
        :param size: The size of the transfer. This value is passed into
            the callbacks

        """
        client.copy_object(CopySource=copy_source, Bucket=bucket, Key=key, **extra_args)
        for callback in callbacks:
            callback(bytes_transferred=size)