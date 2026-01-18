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
class CopyPartTask(Task):
    """Task to upload a part in a multipart copy"""

    def _main(self, client, copy_source, bucket, key, upload_id, part_number, extra_args, callbacks, size):
        """
        :param client: The client to use when calling PutObject
        :param copy_source: The CopySource parameter to use
        :param bucket: The name of the bucket to upload to
        :param key: The name of the key to upload to
        :param upload_id: The id of the upload
        :param part_number: The number representing the part of the multipart
            upload
        :param extra_args: A dictionary of any extra arguments that may be
            used in the upload.
        :param callbacks: List of callbacks to call after copy part
        :param size: The size of the transfer. This value is passed into
            the callbacks

        :rtype: dict
        :returns: A dictionary representing a part::

            {'Etag': etag_value, 'PartNumber': part_number}

            This value can be appended to a list to be used to complete
            the multipart upload.
        """
        response = client.upload_part_copy(CopySource=copy_source, Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=part_number, **extra_args)
        for callback in callbacks:
            callback(bytes_transferred=size)
        etag = response['CopyPartResult']['ETag']
        return {'ETag': etag, 'PartNumber': part_number}