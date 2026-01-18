from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
def _BuildV1Compatibility(self, layer_id, parent, history):
    v1_compatibility = {'id': layer_id}
    if parent:
        v1_compatibility['parent'] = parent
    if 'empty_layer' in history:
        v1_compatibility['throwaway'] = True
    if 'created_by' in history:
        v1_compatibility['container_config'] = {'Cmd': [history['created_by']]}
    for key in ['created', 'comment', 'author']:
        if key in history:
            v1_compatibility[key] = history[key]
    return json.dumps(v1_compatibility, sort_keys=True)