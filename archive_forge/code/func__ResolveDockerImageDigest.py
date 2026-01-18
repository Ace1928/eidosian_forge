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
def _ResolveDockerImageDigest(image):
    """Returns Digest of the given Docker image.

  Lookup registry to get the manifest's digest. If it returns a list of
  manifests, will return the first one.

  Args:
    image: docker_name.Tag or docker_name.Digest, Docker image.

  Returns:
    An str for the digest.
  """
    with v2_2_image_list.FromRegistry(basic_creds=docker_creds.Anonymous(), name=image, transport=transports.GetApitoolsTransport()) as manifest_list:
        if manifest_list.exists():
            return manifest_list.digest()
    with v2_2_image.FromRegistry(basic_creds=docker_creds.Anonymous(), name=image, transport=transports.GetApitoolsTransport(), accepted_mimes=v2_2_docker_http.SUPPORTED_MANIFEST_MIMES) as v2_2_img:
        if v2_2_img.exists():
            return v2_2_img.digest()
        return None