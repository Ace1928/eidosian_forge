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
def _CreateCloudBuild(build_config, client, messages):
    """Create a build in cloud build.

  Args:
    build_config: A cloud build Build message.
    client: The cloud build api client.
    messages: The cloud build api messages module.

  Returns:
    Tuple containing a cloud build build object and the resource reference
    for that build.
  """
    log.debug('submitting build: {0}'.format(repr(build_config)))
    op = client.projects_builds.Create(messages.CloudbuildProjectsBuildsCreateRequest(build=build_config, projectId=properties.VALUES.core.project.Get()))
    json = encoding.MessageToJson(op.metadata)
    build = encoding.JsonToMessage(messages.BuildOperationMetadata, json).build
    build_ref = resources.REGISTRY.Create(collection='cloudbuild.projects.builds', projectId=build.projectId, id=build.id)
    log.CreatedResource(build_ref)
    if build.logUrl:
        log.status.Print('Logs are available at [{0}].'.format(build.logUrl))
    else:
        log.status.Print('Logs are available in the Cloud Console.')
    return (build, build_ref)