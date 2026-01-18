from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import json
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
import httplib2
import six
import six.moves.http_client
class DockerImageList(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for implementations that interact with Docker manifest lists."""

    def digest(self):
        """The digest of the manifest."""
        return docker_digest.SHA256(self.manifest().encode('utf8'))

    def media_type(self):
        """The media type of the manifest."""
        manifest = json.loads(self.manifest())
        return manifest.get('mediaType', docker_http.OCI_IMAGE_INDEX_MIME)

    @abc.abstractmethod
    def manifest(self):
        """The JSON manifest referenced by the tag/digest.

    Returns:
      The raw json manifest
    """

    @abc.abstractmethod
    def resolve_all(self, target=None):
        """Resolves a manifest list to a list of compatible manifests.

    Args:
      target: the platform to check for compatibility. If omitted, the target
          platform defaults to linux/amd64.

    Returns:
      A list of images that can be run on the target platform. The images are
      sorted by their digest.
    """

    def resolve(self, target=None):
        """Resolves a manifest list to a compatible manifest.

    Args:
      target: the platform to check for compatibility. If omitted, the target
          platform defaults to linux/amd64.

    Raises:
      Exception: no manifests were compatible with the target platform.

    Returns:
      An image that can run on the target platform.
    """
        if not target:
            target = Platform()
        images = self.resolve_all(target)
        if not images:
            raise Exception('Could not resolve manifest list to compatible manifest')
        return images[0]

    @abc.abstractmethod
    def __enter__(self):
        """Open the image for reading."""

    @abc.abstractmethod
    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Close the image."""

    @abc.abstractmethod
    def __iter__(self):
        """Iterate over this manifest list's children."""