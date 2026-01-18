from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def CopyFileToGCS(self, local_path, target_obj_ref):
    """Upload a file to the GCS results bucket using the storage API.

    Args:
      local_path: str, the path of the file to upload. File must be on the local
        filesystem.
      target_obj_ref: storage_util.ObjectReference, the path of the file on GCS.

    Returns:
      Object, the storage object that was copied to.

    Raises:
      BucketNotFoundError if the user-specified bucket does not exist.
      UploadError if the file upload is not successful.
      exceptions.BadFileException if the uploaded file size does not match the
          size of the local file.
    """
    file_size = _GetFileSize(local_path)
    src_obj = self.messages.Object(size=file_size)
    mime_type = _GetMimetype(local_path)
    chunksize = self._GetChunkSize()
    upload = transfer.Upload.FromFile(six.ensure_str(local_path), mime_type=mime_type, chunksize=chunksize)
    insert_req = self.messages.StorageObjectsInsertRequest(bucket=target_obj_ref.bucket, name=target_obj_ref.object, object=src_obj)
    gsc_path = '{bucket}/{target_path}'.format(bucket=target_obj_ref.bucket, target_path=target_obj_ref.object)
    log.info('Uploading [{local_file}] to [{gcs}]'.format(local_file=local_path, gcs=gsc_path))
    try:
        response = self.client.objects.Insert(insert_req, upload=upload)
    except api_exceptions.HttpNotFoundError:
        raise BucketNotFoundError('Could not upload file: [{bucket}] bucket does not exist.'.format(bucket=target_obj_ref.bucket))
    except api_exceptions.HttpError as err:
        log.debug('Could not upload file [{local_file}] to [{gcs}]: {e}'.format(local_file=local_path, gcs=gsc_path, e=http_exc.HttpException(err)))
        raise UploadError('{code} Could not upload file [{local_file}] to [{gcs}]: {message}'.format(code=err.status_code, local_file=local_path, gcs=gsc_path, message=http_exc.HttpException(err, error_format='{status_message}')))
    finally:
        upload.stream.close()
    if response.size != file_size:
        log.debug('Response size: {0} bytes, but local file is {1} bytes.'.format(response.size, file_size))
        raise exceptions.BadFileException('Cloud storage upload failure. Uploaded file does not match local file: {0}. Please retry.'.format(local_path))
    return response