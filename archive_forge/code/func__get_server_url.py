import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.compute.types import Provider, NodeState
def _get_server_url(self, uuid):
    return '/v1/servers/%s' % uuid