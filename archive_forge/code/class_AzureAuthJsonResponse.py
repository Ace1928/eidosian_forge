import time
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import urlparse, urlencode, basestring
from libcloud.common.base import BaseDriver, RawResponse, JsonResponse, ConnectionUserAndKey
class AzureAuthJsonResponse(JsonResponse):

    def parse_error(self):
        b = self.parse_body()
        if isinstance(b, basestring):
            return b
        elif isinstance(b, dict) and 'error_description' in b:
            return b['error_description']
        else:
            return str(b)