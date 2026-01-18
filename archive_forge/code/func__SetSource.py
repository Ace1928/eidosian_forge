from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import datetime
import io
import os.path
import shutil
import tarfile
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import staging_bucket_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _SetSource(release_config, source, gcs_source_staging_dir, ignore_file, skaffold_version, location, pipeline_uuid, kubernetes_manifest, cloud_run_manifest, from_run_container, services, skaffold_file, pipeline_obj, hide_logs=False):
    """Set the source for the release config.

  Sets the source for the release config and creates a default Cloud Storage
  bucket with location for staging if gcs-source-staging-dir is not specified.

  Args:
    release_config: a Release message
    source: the location of the source files
    gcs_source_staging_dir: directory in google cloud storage to use for staging
    ignore_file: the ignore file to use
    skaffold_version: version of Skaffold binary
    location: the cloud region for the release
    pipeline_uuid: the unique id of the release's parent pipeline.
    kubernetes_manifest: path to kubernetes manifest (e.g. /home/user/k8.yaml).
      If provided, a Skaffold file will be generated and uploaded to GCS on
      behalf of the customer.
    cloud_run_manifest: path to Cloud Run manifest (e.g.
      /home/user/service.yaml).If provided, a Skaffold file will be generated
      and uploaded to GCS on behalf of the customer.
    from_run_container: the container image (e.g.
      gcr.io/google-containers/nginx@sha256:f49a843c29). If provided, a CloudRun
      manifest file and a Skaffold file will be generated and uploaded to GCS on
      behalf of the customer.
    services: the map from target_id to service_name. This is present only if
      from_run_container is not None.
    skaffold_file: path of the skaffold file relative to the source directory
      that contains the Skaffold file.
    pipeline_obj: the pipeline_obj used for this release.
    hide_logs: whether to show logs, defaults to False

  Returns:
    Modified release_config
  """
    default_gcs_source = False
    default_bucket_name = staging_bucket_util.GetDefaultStagingBucket(pipeline_uuid)
    if gcs_source_staging_dir is None:
        default_gcs_source = True
        gcs_source_staging_dir = _SOURCE_STAGING_TEMPLATE.format(default_bucket_name)
    if not gcs_source_staging_dir.startswith('gs://'):
        raise c_exceptions.InvalidArgumentException(parameter_name='--gcs-source-staging-dir', message=gcs_source_staging_dir)
    gcs_client = storage_api.StorageClient()
    suffix = '.tgz'
    if source.startswith('gs://') or os.path.isfile(source):
        _, suffix = os.path.splitext(source)
    staged_object = '{stamp}-{uuid}{suffix}'.format(stamp=times.GetTimeStampFromDateTime(times.Now()), uuid=uuid.uuid4().hex, suffix=suffix)
    gcs_source_staging_dir = resources.REGISTRY.Parse(gcs_source_staging_dir, collection='storage.objects')
    try:
        gcs_client.CreateBucketIfNotExists(gcs_source_staging_dir.bucket, location=location, check_ownership=default_gcs_source, enable_uniform_level_access=True)
    except storage_api.BucketInWrongProjectError:
        raise c_exceptions.RequiredArgumentException('gcs-source-staging-dir', 'A bucket with name {} already exists and is owned by another project. Specify a bucket using --gcs-source-staging-dir.'.format(default_bucket_name))
    skaffold_is_generated = False
    if gcs_source_staging_dir.object:
        staged_object = gcs_source_staging_dir.object + '/' + staged_object
    gcs_source_staging = resources.REGISTRY.Create(collection='storage.objects', bucket=gcs_source_staging_dir.bucket, object=staged_object)
    if source.startswith('gs://'):
        gcs_source = resources.REGISTRY.Parse(source, collection='storage.objects')
        staged_source_obj = gcs_client.Rewrite(gcs_source, gcs_source_staging)
        release_config.skaffoldConfigUri = 'gs://{bucket}/{object}'.format(bucket=staged_source_obj.bucket, object=staged_source_obj.name)
    elif kubernetes_manifest or cloud_run_manifest or from_run_container:
        skaffold_is_generated = True
        _UploadTarballGeneratedSkaffoldAndManifest(kubernetes_manifest, cloud_run_manifest, from_run_container, services, gcs_client, gcs_source_staging, ignore_file, hide_logs, release_config, pipeline_obj)
    elif os.path.isdir(source):
        _CreateAndUploadTarball(gcs_client, gcs_source_staging, source, ignore_file, hide_logs, release_config)
    elif os.path.isfile(source):
        if not hide_logs:
            log.status.Print('Uploading local file [{src}] to [gs://{bucket}/{object}].'.format(src=source, bucket=gcs_source_staging.bucket, object=gcs_source_staging.object))
        staged_source_obj = gcs_client.CopyFileToGCS(source, gcs_source_staging)
        release_config.skaffoldConfigUri = 'gs://{bucket}/{object}'.format(bucket=staged_source_obj.bucket, object=staged_source_obj.name)
    if skaffold_version:
        release_config.skaffoldVersion = skaffold_version
    release_config = _SetSkaffoldConfigPath(release_config, skaffold_file, skaffold_is_generated)
    return release_config