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
class CloudSigma_1_0_Response(Response):

    def success(self):
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        return 200 <= self.status <= 299

    def parse_body(self):
        if not self.body:
            return self.body
        return str2dicts(self.body)

    def parse_error(self):
        return 'Error: %s' % self.body.replace('errors:', '').strip()