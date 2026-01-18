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
def _perform_action(self, path, action, method='POST', params=None, data=None):
    """
        Perform API action and return response object.
        """
    if params:
        params = params.copy()
    else:
        params = {}
    params['do'] = action
    response = self.connection.request(action=path, method=method, params=params, data=data)
    return response