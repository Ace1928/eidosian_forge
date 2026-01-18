import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigma_2_0_Connection(ConnectionUserAndKey):
    host = API_ENDPOINTS_2_0[DEFAULT_REGION]['host']
    responseCls = CloudSigma_2_0_Response
    api_prefix = '/api/2.0'

    def add_default_headers(self, headers):
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        headers['Authorization'] = 'Basic %s' % base64.b64encode(b('{}:{}'.format(self.user_id, self.key))).decode('utf-8')
        return headers

    def encode_data(self, data):
        data = json.dumps(data)
        return data

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False):
        params = params or {}
        action = self.api_prefix + action
        if method == 'GET':
            params['limit'] = 0
        return super().request(action=action, params=params, data=data, headers=headers, method=method, raw=raw)