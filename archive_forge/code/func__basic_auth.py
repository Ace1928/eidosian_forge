import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
def _basic_auth(self):
    """Constructs basic auth header content string"""
    credentials = b('{}:{}'.format(self.user_id, self.key))
    credentials = base64.b64encode(credentials)
    return 'Basic {}'.format(credentials.decode('ascii'))