import os
import base64
import warnings
from typing import Optional
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import (
from libcloud.common.types import InvalidCredsError
class KubernetesBasicAuthConnection(ConnectionUserAndKey):
    responseCls = KubernetesResponse
    timeout = 60

    def add_default_headers(self, headers):
        """
        Add parameters that are necessary for every request
        If user and password are specified, include a base http auth
        header
        """
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/json'
        if self.user_id and self.key:
            auth_string = b('{}:{}'.format(self.user_id, self.key))
            user_b64 = base64.b64encode(auth_string)
            headers['Authorization'] = 'Basic %s' % user_b64.decode('utf-8')
        return headers