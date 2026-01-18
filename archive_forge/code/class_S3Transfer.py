from botocore.exceptions import ClientError
from botocore.compat import six
from s3transfer.exceptions import RetriesExceededError as \
from s3transfer.manager import TransferConfig as S3TransferConfig
from s3transfer.manager import TransferManager
from s3transfer.futures import NonThreadedExecutor
from s3transfer.subscribers import BaseSubscriber
from s3transfer.utils import OSUtils
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
class S3Transfer(object):
    ALLOWED_DOWNLOAD_ARGS = TransferManager.ALLOWED_DOWNLOAD_ARGS
    ALLOWED_UPLOAD_ARGS = TransferManager.ALLOWED_UPLOAD_ARGS

    def __init__(self, client=None, config=None, osutil=None, manager=None):
        if not client and (not manager):
            raise ValueError('Either a boto3.Client or s3transfer.manager.TransferManager must be provided')
        if manager and any([client, config, osutil]):
            raise ValueError('Manager cannot be provided with client, config, nor osutil. These parameters are mutually exclusive.')
        if config is None:
            config = TransferConfig()
        if osutil is None:
            osutil = OSUtils()
        if manager:
            self._manager = manager
        else:
            self._manager = create_transfer_manager(client, config, osutil)

    def upload_file(self, filename, bucket, key, callback=None, extra_args=None):
        """Upload a file to an S3 object.

        Variants have also been injected into S3 client, Bucket and Object.
        You don't have to use S3Transfer.upload_file() directly.

        .. seealso::
            :py:meth:`S3.Client.upload_file`
            :py:meth:`S3.Client.upload_fileobj`
        """
        if not isinstance(filename, six.string_types):
            raise ValueError('Filename must be a string')
        subscribers = self._get_subscribers(callback)
        future = self._manager.upload(filename, bucket, key, extra_args, subscribers)
        try:
            future.result()
        except ClientError as e:
            raise S3UploadFailedError('Failed to upload %s to %s: %s' % (filename, '/'.join([bucket, key]), e))

    def download_file(self, bucket, key, filename, extra_args=None, callback=None):
        """Download an S3 object to a file.

        Variants have also been injected into S3 client, Bucket and Object.
        You don't have to use S3Transfer.download_file() directly.

        .. seealso::
            :py:meth:`S3.Client.download_file`
            :py:meth:`S3.Client.download_fileobj`
        """
        if not isinstance(filename, six.string_types):
            raise ValueError('Filename must be a string')
        subscribers = self._get_subscribers(callback)
        future = self._manager.download(bucket, key, filename, extra_args, subscribers)
        try:
            future.result()
        except S3TransferRetriesExceededError as e:
            raise RetriesExceededError(e.last_exception)

    def _get_subscribers(self, callback):
        if not callback:
            return None
        return [ProgressCallbackInvoker(callback)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._manager.__exit__(*args)