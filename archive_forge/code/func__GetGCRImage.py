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
def _GetGCRImage(uri):
    """Retrieves information about the given GCR image.

  Args:
    uri: str, The artifact uri.

  Raises:
    ar_exceptions.InvalidInputValueError: If the uri is invalid.

  Returns:
    An Artifact object with metadata of the given artifact.
  """
    location_map = {'us.gcr.io': 'us', 'gcr.io': 'us', 'eu.gcr.io': 'europe', 'asia.gcr.io': 'asia'}
    try:
        docker_digest = gcr_util.GetDigestFromName(uri)
    except gcr_util.InvalidImageNameError as e:
        raise ar_exceptions.InvalidInputValueError('Failed to resolve digest of the GCR image: {}'.format(e))
    project = None
    location = None
    matches = re.match(docker_util.GCR_DOCKER_REPO_REGEX, uri)
    if matches:
        location = location_map[matches.group('repo')]
        project = matches.group('project')
    matches = re.match(docker_util.GCR_DOCKER_DOMAIN_SCOPED_REPO_REGEX, uri)
    if matches:
        location = location_map[matches.group('repo')]
        project = matches.group('project').replace('/', ':', 1)
    if not project or not location:
        raise ar_exceptions.InvalidInputValueError('Failed to parse project and location from the GCR image.')
    return Artifact(resource_uri=docker_digest.__str__(), project=project, location=location, digests={'sha256': docker_digest.digest.replace('sha256:', '')}, artifact_type=ARTIFACT_TYPE_GCR_IMAGE, scheme=_REGISTRY_SCHEME_HTTPS)