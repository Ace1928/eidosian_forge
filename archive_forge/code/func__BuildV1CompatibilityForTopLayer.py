from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
def _BuildV1CompatibilityForTopLayer(self, layer_id, parent, history, config):
    v1_compatibility = {'id': layer_id}
    if parent:
        v1_compatibility['parent'] = parent
    if 'empty_layer' in history:
        v1_compatibility['throwaway'] = True
    for key in ['architecture', 'container', 'docker_version', 'os', 'config', 'container_config', 'created']:
        if key in config:
            v1_compatibility[key] = config[key]
    return json.dumps(v1_compatibility, sort_keys=True)