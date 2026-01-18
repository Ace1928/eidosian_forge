from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
def _ProcessImage(self):
    """Constructs schema 1 manifest from schema 2 manifest and config file."""
    manifest_schema2 = json.loads(self._v2_2_image.manifest())
    raw_config = self._v2_2_image.config_file()
    config = json.loads(raw_config)
    parent = ''
    histories = config.get('history', {})
    layer_count = len(histories)
    v2_layer_index = 0
    layers = manifest_schema2.get('layers', {})
    fs_layers = []
    v1_histories = []
    for v1_layer_index, history in enumerate(histories):
        digest, media_type, v2_layer_index = self._GetSchema1LayerDigest(history, layers, v1_layer_index, v2_layer_index)
        if v1_layer_index != layer_count - 1:
            layer_id = self._GenerateV1LayerId(digest, parent)
            v1_compatibility = self._BuildV1Compatibility(layer_id, parent, history)
        else:
            layer_id = self._GenerateV1LayerId(digest, parent, raw_config)
            v1_compatibility = self._BuildV1CompatibilityForTopLayer(layer_id, parent, history, config)
        parent = layer_id
        fs_layers = [{'blobSum': digest, 'mediaType': media_type}] + fs_layers
        v1_histories = [{'v1Compatibility': v1_compatibility}] + v1_histories
    manifest_schema1 = {'schemaVersion': 1, 'name': 'unused', 'tag': 'unused', 'fsLayers': fs_layers, 'history': v1_histories}
    if 'architecture' in config:
        manifest_schema1['architecture'] = config['architecture']
    self._manifest = v2_util.Sign(json.dumps(manifest_schema1, sort_keys=True))