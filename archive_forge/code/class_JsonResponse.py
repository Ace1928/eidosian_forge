import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
class JsonResponse(Response):
    """
    A Base JSON Response class to derive from.
    """

    def parse_body(self):
        if len(self.body) == 0 and (not self.parse_zero_length_body):
            return self.body
        try:
            body = json.loads(self.body)
        except Exception:
            raise MalformedResponseError('Failed to parse JSON', body=self.body, driver=self.connection.driver)
        return body
    parse_error = parse_body