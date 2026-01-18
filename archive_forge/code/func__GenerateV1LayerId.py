from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
def _GenerateV1LayerId(self, digest, parent, raw_config=None):
    parts = digest.rsplit(':', 1)
    if len(parts) != 2:
        raise BadDigestException('Invalid Digest: %s, must be in algorithm : blobSumHex format.' % digest)
    data = parts[1] + ' ' + parent
    if raw_config:
        data += ' ' + raw_config
    return docker_digest.SHA256(data.encode('utf8'), '')