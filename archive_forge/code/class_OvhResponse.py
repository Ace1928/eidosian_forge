import time
import hashlib
from typing import List
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.utils.connection import get_response_object
class OvhResponse(JsonResponse):

    def parse_error(self):
        response = super().parse_body()
        response = response or {}
        if response.get('errorCode', None) == 'INVALID_SIGNATURE':
            raise InvalidCredsError('Signature validation failed, probably using invalid credentials')
        return self.body