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
class FromList(DockerImageList):
    """This synthesizes a Manifest List from a list of images."""

    def __init__(self, images):
        self._images = images
        super(FromList, self).__init__()

    def manifest(self):
        list_body = {'mediaType': docker_http.MANIFEST_LIST_MIME, 'schemaVersion': 2, 'manifests': []}
        for platform, manifest in self._images:
            manifest_body = {'digest': manifest.digest(), 'mediaType': manifest.media_type(), 'size': len(manifest.manifest())}
            if platform:
                manifest_body['platform'] = dict(platform)
            list_body['manifests'].append(manifest_body)
        return json.dumps(list_body, sort_keys=True)

    def resolve_all(self, target=None):
        """Resolves a manifest list to a list of compatible manifests.

    Args:
      target: the platform to check for compatibility. If omitted, the target
          platform defaults to linux/amd64.

    Returns:
      A list of images that can be run on the target platform.
    """
        target = target or Platform()
        results = []
        for platform, image in self._images:
            if isinstance(image, DockerImageList):
                with image:
                    results.extend(image.resolve_all(target))
            elif target.can_run(platform):
                results.append(image)
        dgst_img_dict = {img.digest(): img for img in results}
        results = []
        return [dgst_img_dict[dgst] for dgst in sorted(dgst_img_dict.keys())]

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass

    def __iter__(self):
        return iter(self._images)