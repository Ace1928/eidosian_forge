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
def _tag_or_digest(name):
    if isinstance(name, docker_name.Tag):
        return name.tag
    else:
        assert isinstance(name, docker_name.Digest)
        return name.digest