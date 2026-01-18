from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_creds as v1_creds
from containerregistry.client.v1 import docker_http
from containerregistry.client.v1 import docker_image
import httplib2
import six.moves.http_client
def _put_layer(self, image, layer_id):
    """Upload the aufs tarball for a single layer."""
    docker_http.Request(self._transport, '{scheme}://{endpoint}/v1/images/{layer}/layer'.format(scheme=docker_http.Scheme(self._endpoint), endpoint=self._endpoint, layer=layer_id), self._token_creds, accepted_codes=[six.moves.http_client.OK], body=image.layer(layer_id), content_type='application/octet-stream')