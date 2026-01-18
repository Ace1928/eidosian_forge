import json
import time
import datetime
from libcloud.utils.py3 import basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
class KamateraConnection(ConnectionUserAndKey):
    """
    Connection class for KamateraDriver
    """
    host = 'cloudcli.cloudwm.com'
    responseCls = KamateraResponse

    def add_default_headers(self, headers):
        """Adds headers that are needed for all requests"""
        headers['AuthClientId'] = self.user_id
        headers['AuthSecret'] = self.key
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        return headers