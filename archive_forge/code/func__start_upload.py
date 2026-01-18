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
def _start_upload(self, digest, mount=None):
    """POST to begin the upload process with optional cross-repo mount param."""
    if not mount:
        url = '{base_url}/blobs/uploads/'.format(base_url=self._base_url())
        accepted_codes = [six.moves.http_client.ACCEPTED]
    else:
        mount_from = '&'.join(['from=' + six.moves.urllib.parse.quote(repo.repository, '') for repo in self._mount])
        url = '{base_url}/blobs/uploads/?mount={digest}&{mount_from}'.format(base_url=self._base_url(), digest=digest, mount_from=mount_from)
        accepted_codes = [six.moves.http_client.CREATED, six.moves.http_client.ACCEPTED]
    resp, unused_content = self._transport.Request(url, method='POST', body=None, accepted_codes=accepted_codes)
    return (resp.status == six.moves.http_client.CREATED, resp.get('location'))