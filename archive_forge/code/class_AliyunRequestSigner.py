import sys
import hmac
import time
import uuid
import base64
import hashlib
from libcloud.utils.py3 import ET, b, u, urlquote
from libcloud.utils.xml import findtext
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import MalformedResponseError
class AliyunRequestSigner:
    """
    Class handles signing the outgoing Aliyun requests.
    """

    def __init__(self, access_key, access_secret, version):
        """
        :param access_key: Access key.
        :type access_key: ``str``

        :param access_secret: Access secret.
        :type access_secret: ``str``

        :param version: API version.
        :type version: ``str``
        """
        self.access_key = access_key
        self.access_secret = access_secret
        self.version = version

    def get_request_params(self, params, method='GET', path='/'):
        return params

    def get_request_headers(self, params, headers, method='GET', path='/'):
        return (params, headers)