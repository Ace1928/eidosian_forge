from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _UploadSourceToGCS(source, stage_bucket, deployment_short_name, location, ignore_file):
    """Uploads local content to GCS.

  This will ensure that the source and destination exist before triggering the
  upload.

  Args:
    source: string, a local path.
    stage_bucket: optional string. When not provided, the default staging bucket
      will be used (see GetDefaultStagingBucket). This string is of the format
      "gs://bucket-name/". An "im_source_staging" object will be created under
      this bucket, and any uploaded artifacts will be stored there.
    deployment_short_name: short name of the deployment.
    location: location of the deployment.
    ignore_file: string, a path to a gcloudignore file.

  Returns:
    A string in the format "gs://path/to/resulting/upload".

  Raises:
    RequiredArgumentException: if stage-bucket is owned by another project.
    BadFileException: if the source doesn't exist or isn't a directory.
  """
    gcs_client = storage_api.StorageClient()
    if stage_bucket is None:
        used_default_bucket_name = True
        gcs_source_staging_dir = staging_bucket_util.DefaultGCSStagingDir(deployment_short_name, location)
    else:
        used_default_bucket_name = False
        gcs_source_staging_dir = '{0}{1}/{2}/{3}'.format(stage_bucket, staging_bucket_util.STAGING_DIR, location, deployment_short_name)
    gcs_source_staging_dir_ref = resources.REGISTRY.Parse(gcs_source_staging_dir, collection='storage.objects')
    try:
        gcs_client.CreateBucketIfNotExists(gcs_source_staging_dir_ref.bucket, check_ownership=used_default_bucket_name)
    except storage_api.BucketInWrongProjectError:
        raise c_exceptions.RequiredArgumentException('stage-bucket', 'A bucket with name {} already exists and is owned by another project. Specify a bucket using --stage-bucket.'.format(gcs_source_staging_dir_ref.bucket))
    staged_object = '{stamp}-{uuid}'.format(stamp=times.GetTimeStampFromDateTime(times.Now()), uuid=uuid.uuid4().hex)
    if gcs_source_staging_dir_ref.object:
        staged_object = gcs_source_staging_dir_ref.object + '/' + staged_object
    gcs_source_staging = resources.REGISTRY.Create(collection='storage.objects', bucket=gcs_source_staging_dir_ref.bucket, object=staged_object)
    if not os.path.exists(source):
        raise c_exceptions.BadFileException('could not find source [{}]'.format(source))
    if not os.path.isdir(source):
        raise c_exceptions.BadFileException('source is not a directory [{}]'.format(source))
    _UploadSourceDirToGCS(gcs_client, source, gcs_source_staging, ignore_file)
    upload_bucket = 'gs://{0}/{1}'.format(gcs_source_staging.bucket, gcs_source_staging.object)
    return upload_bucket