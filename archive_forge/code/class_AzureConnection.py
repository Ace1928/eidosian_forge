import os
import copy
import hmac
import time
import base64
from hashlib import sha256
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import ET, b, httplib, urlparse, urlencode, basestring
from libcloud.utils.xml import fixxpath
from libcloud.common.base import (
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.common.azure_arm import AzureAuthJsonResponse, publicEnvironments
class AzureConnection(ConnectionUserAndKey):
    """
    Represents a single connection to Azure
    """
    responseCls = AzureResponse
    rawResponseCls = AzureRawResponse
    API_VERSION = '2012-02-12'

    def add_default_params(self, params):
        return params

    def pre_connect_hook(self, params, headers):
        headers = copy.deepcopy(headers)
        headers['x-ms-date'] = time.strftime(AZURE_TIME_FORMAT, time.gmtime())
        headers['x-ms-version'] = self.API_VERSION
        headers['Authorization'] = self._get_azure_auth_signature(method=self.method, headers=headers, params=params, account=self.user_id, secret_key=self.key, path=self.action)
        headers.pop('Host', None)
        return (params, headers)

    def _get_azure_auth_signature(self, method, headers, params, account, secret_key, path='/'):
        """
        Signature = Base64( HMAC-SHA1( YourSecretAccessKeyID,
                            UTF-8-Encoding-Of( StringToSign ) ) ) );

        StringToSign = HTTP-VERB + "
" +
            Content-Encoding + "
" +
            Content-Language + "
" +
            Content-Length + "
" +
            Content-MD5 + "
" +
            Content-Type + "
" +
            Date + "
" +
            If-Modified-Since + "
" +
            If-Match + "
" +
            If-None-Match + "
" +
            If-Unmodified-Since + "
" +
            Range + "
" +
            CanonicalizedHeaders +
            CanonicalizedResource;
        """
        xms_header_values = []
        param_list = []
        headers_copy = {}
        for header, value in headers.items():
            header = header.lower()
            value = str(value).strip()
            if header.startswith('x-ms-'):
                xms_header_values.append((header, value))
            else:
                headers_copy[header] = value
        special_header_values = self._format_special_header_values(headers_copy, method)
        values_to_sign = [method] + special_header_values
        xms_header_values.sort()
        for header, value in xms_header_values:
            values_to_sign.append('{}:{}'.format(header, value))
        values_to_sign.append('/{}{}'.format(account, path))
        for key, value in params.items():
            param_list.append((key.lower(), str(value).strip()))
        param_list.sort()
        for key, value in param_list:
            values_to_sign.append('{}:{}'.format(key, value))
        string_to_sign = b('\n'.join(values_to_sign))
        secret_key = b(secret_key)
        b64_hmac = base64.b64encode(hmac.new(secret_key, string_to_sign, digestmod=sha256).digest())
        return 'SharedKey {}:{}'.format(self.user_id, b64_hmac.decode('utf-8'))

    def _format_special_header_values(self, headers, method):
        is_change = method not in ('GET', 'HEAD')
        is_old_api = self.API_VERSION <= '2014-02-14'
        special_header_keys = ['content-encoding', 'content-language', 'content-length', 'content-md5', 'content-type', 'date', 'if-modified-since', 'if-match', 'if-none-match', 'if-unmodified-since', 'range']
        special_header_values = []
        for header in special_header_keys:
            header = header.lower()
            if header in headers:
                special_header_values.append(headers[header])
            elif header == 'content-length' and is_change and is_old_api:
                special_header_values.append('0')
            else:
                special_header_values.append('')
        return special_header_values