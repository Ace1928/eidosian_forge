import copy
import hmac
import base64
import hashlib
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.compute.types import InvalidCredsError
def _sync_request(self, command, action=None, params=None, data=None, headers=None, method='GET'):
    return self.connection._sync_request(command=command, action=action, params=params, data=data, headers=headers, method=method)