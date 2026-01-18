from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import scheduler
from googlecloudsdk.api_lib import tasks
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import deploy_app_command_util
from googlecloudsdk.api_lib.app import deploy_command_util
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.datastore import index_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.tasks import app_deploy_migration_util
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import create_util
from googlecloudsdk.command_lib.app import deployables
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.command_lib.app import source_files_util
from googlecloudsdk.command_lib.app import staging
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _PossiblyUploadFiles(self, image, service_info, upload_dir, source_files, code_bucket_ref, flex_image_build_option):
    """Uploads files for this deployment is required for this service.

    Uploads if flex_image_build_option is FlexImageBuildOptions.ON_SERVER,
    or if the deployment is non-hermetic and the image is not provided.

    Args:
      image: str or None, the URL for the Docker image to be deployed (if image
        already exists).
      service_info: yaml_parsing.ServiceYamlInfo, service configuration to be
        deployed
      upload_dir: str, path to the service's upload directory
      source_files: [str], relative paths to upload.
      code_bucket_ref: cloud_storage.BucketReference where the service's files
        have been uploaded
      flex_image_build_option: FlexImageBuildOptions, whether a flex deployment
        should upload files so that the server can build the image or build the
        image on client or build the image on client using the buildpacks.

    Returns:
      Dictionary mapping source files to Google Cloud Storage locations.

    Raises:
      RequiredFileMissingError: if a required file is not uploaded.
    """
    manifest = None
    if not image and (flex_image_build_option == FlexImageBuildOptions.ON_SERVER or not service_info.is_hermetic):
        if service_info.env == env.FLEX and (not _AppYamlInSourceFiles(source_files, service_info.GetAppYamlBasename())):
            raise RequiredFileMissingError(service_info.GetAppYamlBasename())
        limit = None
        if service_info.env == env.STANDARD and service_info.runtime in _RUNTIMES_WITH_FILE_SIZE_LIMITS:
            limit = _MAX_FILE_SIZE_STANDARD
        manifest = deploy_app_command_util.CopyFilesToCodeBucket(upload_dir, source_files, code_bucket_ref, max_file_size=limit)
    return manifest