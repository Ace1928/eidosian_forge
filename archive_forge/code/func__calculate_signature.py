import hmac
import time
import hashlib
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Container, StorageDriver
def _calculate_signature(self, user_id, method, params, path, timestamp, key):
    if params:
        uri_path = path + '?' + urlencode(params)
    else:
        uri_path = path
    string_to_sign = [user_id, method, str(timestamp), uri_path]
    string_to_sign = '\n'.join(string_to_sign)
    hmac_value = hmac.new(key, string_to_sign, hashlib.sha256)
    return hmac_value.hexdigest()