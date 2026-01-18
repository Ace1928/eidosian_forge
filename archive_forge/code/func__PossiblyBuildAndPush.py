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
def _PossiblyBuildAndPush(self, new_version, service, upload_dir, source_files, image, code_bucket_ref, gcr_domain, flex_image_build_option):
    """Builds and Pushes the Docker image if necessary for this service.

    Args:
      new_version: version_util.Version describing where to deploy the service
      service: yaml_parsing.ServiceYamlInfo, service configuration to be
        deployed
      upload_dir: str, path to the service's upload directory
      source_files: [str], relative paths to upload.
      image: str or None, the URL for the Docker image to be deployed (if image
        already exists).
      code_bucket_ref: cloud_storage.BucketReference where the service's files
        have been uploaded
      gcr_domain: str, Cloud Registry domain, determines the physical location
        of the image. E.g. `us.gcr.io`.
      flex_image_build_option: FlexImageBuildOptions, whether a flex deployment
        should upload files so that the server can build the image or build the
        image on client or build the image on client using the buildpacks.

    Returns:
      BuildArtifact, a wrapper which contains either the build ID for
        an in-progress build, or the name of the container image for a serial
        build. Possibly None if the service does not require an image.
    Raises:
      RequiredFileMissingError: if a required file is not uploaded.
    """
    build = None
    if image:
        if service.RequiresImage() and service.parsed.skip_files.regex:
            log.warning('Deployment of service [{0}] will ignore the skip_files field in the configuration file, because the image has already been built.'.format(new_version.service))
        return app_cloud_build.BuildArtifact.MakeImageArtifact(image)
    elif service.RequiresImage():
        if not _AppYamlInSourceFiles(source_files, service.GetAppYamlBasename()):
            raise RequiredFileMissingError(service.GetAppYamlBasename())
        if flex_image_build_option == FlexImageBuildOptions.ON_SERVER:
            cloud_build_options = {'appYamlPath': service.GetAppYamlBasename()}
            timeout = properties.VALUES.app.cloud_build_timeout.Get()
            if timeout:
                build_timeout = int(times.ParseDuration(timeout, default_suffix='s').total_seconds)
                cloud_build_options['cloudBuildTimeout'] = six.text_type(build_timeout) + 's'
            build = app_cloud_build.BuildArtifact.MakeBuildOptionsArtifact(cloud_build_options)
        else:
            build = deploy_command_util.BuildAndPushDockerImage(new_version.project, service, upload_dir, source_files, new_version.id, code_bucket_ref, gcr_domain, self.deploy_options.runtime_builder_strategy, self.deploy_options.parallel_build, flex_image_build_option == FlexImageBuildOptions.BUILDPACK_ON_CLIENT)
    return build