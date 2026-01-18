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
def _put_images(self):
    """Close the session by putting to the .../images endpoint."""
    docker_http.Request(self._transport, '{scheme}://{registry}/v1/repositories/{repository}/images'.format(scheme=docker_http.Scheme(self._name.registry), registry=self._name.registry, repository=self._name.repository), self._basic_creds, accepted_codes=[six.moves.http_client.NO_CONTENT], body=b'[]')