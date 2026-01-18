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
def _CreateAndUploadTarball(gcs_client, gcs_source_staging, source, ignore_file, hide_logs, release_config, print_skaffold_config=False):
    """Creates a local tarball and uploads it to GCS.

     After creating and uploading the tarball, this sets the Skaffold config URI
     in the release config.

  Args:
    gcs_client: client for Google Cloud Storage API.
    gcs_source_staging: directory in Google cloud storage to use for staging
    source: the location of the source files
    ignore_file: the ignore file to use
    hide_logs: whether to show logs, defaults to False
    release_config: release configuration
    print_skaffold_config: if true, the Cloud Storage URI of tar.gz archive
      containing Skaffold configuration will be printed, defaults to False.
  """
    source_snapshot = snapshot.Snapshot(source, ignore_file=ignore_file)
    size_str = resource_transform.TransformSize(source_snapshot.uncompressed_size)
    if not hide_logs:
        log.status.Print('Creating temporary archive of {num_files} file(s) totalling {size} before compression.'.format(num_files=len(source_snapshot.files), size=size_str))
    staged_source_obj = source_snapshot.CopyArchiveToGCS(gcs_client, gcs_source_staging, ignore_file=ignore_file, hide_logs=hide_logs)
    release_config.skaffoldConfigUri = 'gs://{bucket}/{object}'.format(bucket=staged_source_obj.bucket, object=staged_source_obj.name)
    if print_skaffold_config:
        log.status.Print('Generated Skaffold file can be found here: {config_uri}'.format(config_uri=release_config.skaffoldConfigUri))