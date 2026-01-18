from __future__ import (absolute_import, division, print_function)
import os
import os.path
import socket as pysocket
import struct
from ansible.module_utils.six import PY2
from ansible_collections.community.docker.plugins.module_utils._api.utils import socket as docker_socket
from ansible_collections.community.docker.plugins.module_utils.socket_helper import (
class DockerSocketHandlerModule(DockerSocketHandlerBase):

    def __init__(self, sock, module, selectors):
        super(DockerSocketHandlerModule, self).__init__(sock, selectors, module.debug)