from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import concurrent.futures
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list as image_list
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def _remote_tag_digest(self, image):
    """Check the remote for the given manifest by digest."""
    resp, unused_content = self._transport.Request('{base_url}/manifests/{tag}'.format(base_url=self._base_url(), tag=self._name.tag), method='GET', accepted_codes=[six.moves.http_client.OK, six.moves.http_client.NOT_FOUND], accepted_mimes=[image.media_type()])
    if resp.status == six.moves.http_client.NOT_FOUND:
        return None
    return resp.get('docker-content-digest')