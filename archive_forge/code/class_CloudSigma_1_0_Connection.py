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
class CloudSigma_1_0_Connection(ConnectionUserAndKey):
    host = API_ENDPOINTS_1_0[DEFAULT_REGION]['host']
    responseCls = CloudSigma_1_0_Response

    def add_default_headers(self, headers):
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        headers['Authorization'] = 'Basic %s' % base64.b64encode(b('{}:{}'.format(self.user_id, self.key))).decode('utf-8')
        return headers