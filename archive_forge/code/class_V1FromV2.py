from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v2 import docker_digest
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util
from six.moves import zip  # pylint: disable=redefined-builtin
class V1FromV2(v1_image.DockerImage):
    """This compatibility interface serves the v1 interface from a v2 image."""

    def __init__(self, v2_img):
        """Constructor.

    Args:
      v2_img: a v2 DockerImage on which __enter__ has already been called.
    """
        self._v2_image = v2_img
        self._ComputeLayerMapping()

    def _ComputeLayerMapping(self):
        """Parse the v2 manifest and extract indices to efficiently answer v1 apis.

    This reads the v2 manifest, corrolating the v1 compatibility and v2 fsLayer
    arrays and creating three indices for efficiently answering v1 queries:
      self._v1_to_v2: dict, maps from v1 layer id to v2 digest
      self._v1_json: dict, maps from v1 layer id to v1 json
      self._v1_ancestry: list, the order of the v1 layers
    """
        raw_manifest = self._v2_image.manifest()
        manifest = json.loads(raw_manifest)
        v2_ancestry = [fs_layer['blobSum'] for fs_layer in manifest['fsLayers']]
        v1_jsons = [v1_layer['v1Compatibility'] for v1_layer in manifest['history']]

        def ExtractId(v1_json):
            v1_metadata = json.loads(v1_json)
            return v1_metadata['id']
        self._v1_to_v2 = {}
        self._v1_json = {}
        self._v1_ancestry = []
        for v1_json, v2_digest in zip(v1_jsons, v2_ancestry):
            v1_id = ExtractId(v1_json)
            if v1_id in self._v1_to_v2:
                assert self._v1_to_v2[v1_id] == v2_digest
                assert self._v1_json[v1_id] == v1_json
                continue
            self._v1_to_v2[v1_id] = v2_digest
            self._v1_json[v1_id] = v1_json
            self._v1_ancestry.append(v1_id)

    def top(self):
        """Override."""
        return self._v1_ancestry[0]

    def repositories(self):
        """Override."""
        pass

    def parent(self, layer_id):
        """Override."""
        ancestry = self.ancestry(layer_id)
        if len(ancestry) == 1:
            return None
        return ancestry[1]

    def json(self, layer_id):
        """Override."""
        return self._v1_json.get(layer_id, '{}')

    def uncompressed_layer(self, layer_id):
        """Override."""
        v2_digest = self._v1_to_v2.get(layer_id)
        return self._v2_image.uncompressed_blob(v2_digest)

    def layer(self, layer_id):
        """Override."""
        v2_digest = self._v1_to_v2.get(layer_id)
        return self._v2_image.blob(v2_digest)

    def diff_id(self, digest):
        """Override."""
        return self._v2_image.diff_id(self._v1_to_v2.get(digest))

    def ancestry(self, layer_id):
        """Override."""
        index = self._v1_ancestry.index(layer_id)
        return self._v1_ancestry[index:]

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass