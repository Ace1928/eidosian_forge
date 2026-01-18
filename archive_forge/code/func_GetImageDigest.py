from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def GetImageDigest(artifact_url):
    """Returns the digest of an image given its url.

  Args:
    artifact_url: An image url. e.g. "https://gcr.io/foo/bar@sha256:123"

  Returns:
    The image digest. e.g. "sha256:123"
  """
    url_without_scheme = ReplaceImageUrlScheme(artifact_url, scheme='')
    try:
        digest = docker_name.Digest(url_without_scheme)
    except docker_name.BadNameException as e:
        raise BadImageUrlError(e)
    return digest.digest