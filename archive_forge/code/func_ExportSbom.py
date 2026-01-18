from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def ExportSbom(args):
    """Export SBOM files for a given AR image.

  Args:
    args: User input arguments.
  """
    if not args.uri:
        raise ar_exceptions.InvalidInputValueError('--uri is required.')
    uri = _RemovePrefix(args.uri, 'https://')
    if docker_util.IsARDockerImage(uri):
        artifact = _GetARDockerImage(uri)
    elif docker_util.IsGCRImage(uri):
        artifact = _GetGCRImage(uri)
        messages = ar_requests.GetMessages()
        settings = ar_requests.GetProjectSettings(artifact.project)
        if settings.legacyRedirectionState != messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED:
            raise ar_exceptions.InvalidInputValueError('This command only supports Artifact Registry. You can enable redirection to use gcr.io repositories in Artifact Registry.')
    else:
        raise ar_exceptions.InvalidInputValueError('{} is not an Artifact Registry image.'.format(uri))
    project = util.GetProject(args)
    if artifact.project:
        project = artifact.project
    resp = ca_requests.ExportSbomV1beta1(project, 'https://{}'.format(artifact.resource_uri))
    log.status.Print('Exporting the SBOM file for resource {}. Discovery occurrence ID: {}'.format(artifact.resource_uri, resp.discoveryOccurrenceId))