import os
import hmac
import time
import base64
import codecs
from hashlib import sha1
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class OSSConnection(ConnectionUserAndKey):
    """
    Represents a single connection to the Aliyun OSS Endpoint
    """
    _domain = 'aliyuncs.com'
    _default_location = 'oss'
    responseCls = OSSResponse
    rawResponseCls = OSSRawResponse

    @staticmethod
    def _get_auth_signature(method, headers, params, expires, secret_key, path, vendor_prefix):
        """
        Signature = base64(hmac-sha1(AccessKeySecret,
          VERB + "
"
          + CONTENT-MD5 + "
"
          + CONTENT-TYPE + "
"
          + EXPIRES + "
"
          + CanonicalizedOSSHeaders
          + CanonicalizedResource))
        """
        special_headers = {'content-md5': '', 'content-type': '', 'expires': ''}
        vendor_headers = {}
        for key, value in list(headers.items()):
            key_lower = key.lower()
            if key_lower in special_headers:
                special_headers[key_lower] = value.strip()
            elif key_lower.startswith(vendor_prefix):
                vendor_headers[key_lower] = value.strip()
        if expires:
            special_headers['expires'] = str(expires)
        buf = [method]
        for _, value in sorted(special_headers.items()):
            buf.append(value)
        string_to_sign = '\n'.join(buf)
        buf = []
        for key, value in sorted(vendor_headers.items()):
            buf.append('{}:{}'.format(key, value))
        header_string = '\n'.join(buf)
        values_to_sign = []
        for value in [string_to_sign, header_string, path]:
            if value:
                values_to_sign.append(value)
        string_to_sign = '\n'.join(values_to_sign)
        b64_hmac = base64.b64encode(hmac.new(b(secret_key), b(string_to_sign), digestmod=sha1).digest())
        return b64_hmac

    @staticmethod
    def _get_expires(params):
        """
        Get expires timeout seconds from parameters.
        """
        expires = None
        if 'expires' in params:
            expires = params['expires']
        elif 'Expires' in params:
            expires = params['Expires']
        if expires:
            try:
                return int(expires)
            except Exception:
                pass
        return int(time.time()) + EXPIRATION_SECONDS

    def add_default_params(self, params):
        expires_at = self._get_expires(params)
        expires = str(expires_at)
        params['OSSAccessKeyId'] = self.user_id
        params['Expires'] = expires
        return params

    def add_default_headers(self, headers):
        headers['Date'] = time.strftime(GMT_TIME_FORMAT, time.gmtime())
        return headers

    def pre_connect_hook(self, params, headers):
        if self._container:
            path = '/{}{}'.format(self._container.name, self.action)
        else:
            path = self.action
        params['Signature'] = self._get_auth_signature(method=self.method, headers=headers, params=params, expires=params['Expires'], secret_key=self.key, path=path, vendor_prefix=self.driver.http_vendor_prefix)
        return (params, headers)

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False, container=None):
        self.host = '{}.{}'.format(self._default_location, self._domain)
        self._container = container
        if container and container.name:
            if 'location' in container.extra:
                self.host = '{}.{}.{}'.format(container.name, container.extra['location'], self._domain)
            else:
                self.host = '{}.{}'.format(container.name, self.host)
        return super().request(action=action, params=params, data=data, headers=headers, method=method, raw=raw)