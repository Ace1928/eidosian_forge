import copy
import logging
from s3transfer.utils import get_callbacks
class CreateMultipartUploadTask(Task):
    """Task to initiate a multipart upload"""

    def _main(self, client, bucket, key, extra_args):
        """
        :param client: The client to use when calling CreateMultipartUpload
        :param bucket: The name of the bucket to upload to
        :param key: The name of the key to upload to
        :param extra_args: A dictionary of any extra arguments that may be
            used in the intialization.

        :returns: The upload id of the multipart upload
        """
        response = client.create_multipart_upload(Bucket=bucket, Key=key, **extra_args)
        upload_id = response['UploadId']
        self._transfer_coordinator.add_failure_cleanup(client.abort_multipart_upload, Bucket=bucket, Key=key, UploadId=upload_id)
        return upload_id