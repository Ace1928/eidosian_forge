from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def RunImageCloudBuild(args, builder, builder_args, tags, output_filter, cloudbuild_service_account_roles, compute_service_account_roles, build_region=None):
    """Run a build related to image on Google Cloud Builder.

  Args:
    args: An argparse namespace. All the arguments that were provided to this
      command invocation.
    builder: A path to builder image.
    builder_args: A list of key-value pairs to pass to builder.
    tags: A list of strings for adding tags to the Argo build.
    output_filter: A list of strings indicating what lines from the log should
      be output. Only lines that start with one of the strings in output_filter
      will be displayed.
    cloudbuild_service_account_roles: roles required for cloudbuild service
      account.
    compute_service_account_roles: roles required for compute service account.
    build_region: Region to run Cloud Build in.

  Returns:
    A build object that either streams the output or is displayed as a
    link to the build.

  Raises:
    FailedBuildException: If the build is completed and not 'SUCCESS'.
  """
    project_id = projects_util.ParseProject(properties.VALUES.core.project.GetOrFail())
    _CheckIamPermissions(project_id, frozenset(cloudbuild_service_account_roles), frozenset(compute_service_account_roles), args.cloudbuild_service_account if 'cloudbuild_service_account' in args else '', args.compute_service_account if 'compute_service_account' in args else '')
    return _RunCloudBuild(args, builder, builder_args, ['gce-daisy'] + tags, output_filter, args.log_location, build_region=build_region)