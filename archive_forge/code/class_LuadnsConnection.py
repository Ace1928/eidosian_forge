import base64
from typing import Dict, List
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
class LuadnsConnection(ConnectionUserAndKey):
    host = API_HOST
    responseCls = LuadnsResponse

    def add_default_headers(self, headers):
        b64string = b('{}:{}'.format(self.user_id, self.key))
        encoded = base64.b64encode(b64string).decode('utf-8')
        authorization = 'Basic ' + encoded
        headers['Accept'] = 'application/json'
        headers['Authorization'] = authorization
        return headers