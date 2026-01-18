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
def _GetDockerImage(uri):
    """Retrieves information about the given docker image.

  Args:
    uri: str, The artifact uri.

  Raises:
    ar_exceptions.InvalidInputValueError: If the artifact is with tag, and it
    can not be resolved by querying the docker http APIs.

  Returns:
    An Artifact object with metadata of the given artifact.
  """
    try:
        image_digest = docker_name.from_string(uri)
        if isinstance(image_digest, docker_name.Digest):
            return Artifact(resource_uri=uri, digests={'sha256': image_digest.digest.replace('sha256:', '')}, artifact_type=ARTIFACT_TYPE_OTHER, project=None, location=None, scheme=None)
    except (docker_name.BadNameException,) as e:
        raise ar_exceptions.InvalidInputValueError('Failed to resolve {0}: {1}'.format(uri, str(e)))
    image_uri = uri
    if ':' not in uri:
        image_uri = uri + ':latest'
    image_tag = docker_name.Tag(name=image_uri)
    scheme = v2_2_docker_http.Scheme(image_tag.registry)
    try:
        digest = _ResolveDockerImageDigest(image_tag)
    except (v2_2_docker_http.V2DiagnosticException, requests.exceptions.InvalidURL, v2_2_docker_http.BadStateException) as e:
        raise ar_exceptions.InvalidInputValueError('Failed to resolve {0}: {1}'.format(uri, str(e)))
    if not digest:
        raise ar_exceptions.InvalidInputValueError('Failed to resolve {0}.'.format(uri))
    resource_uri = '{registry}/{repo}@{digest}'.format(registry=image_tag.registry, repo=image_tag.repository, digest=digest)
    return Artifact(resource_uri=resource_uri, digests={'sha256': digest.replace('sha256:', '')}, artifact_type=ARTIFACT_TYPE_OTHER, project=None, location=None, scheme=scheme)