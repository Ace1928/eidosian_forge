import base64
import hashlib
from libcloud.utils.py3 import b, next, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.utils.escape import sanitize_object_name
from libcloud.storage.types import ObjectDoesNotExistError, ContainerDoesNotExistError
from libcloud.storage.providers import Provider
class BackblazeB2Connection(ConnectionUserAndKey):
    host = None
    secure = True
    responseCls = BackblazeB2Response
    authCls = BackblazeB2AuthConnection

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auth_conn = self.authCls(*args, **kwargs)

    def download_request(self, action, params=None):
        auth_conn = self._auth_conn.authenticate()
        self._set_host(auth_conn.download_host)
        action = '/file/' + action
        method = 'GET'
        raw = True
        response = self._request(auth_conn=auth_conn, action=action, params=params, method=method, raw=raw)
        return response

    def upload_request(self, action, headers, upload_host, auth_token, data):
        auth_conn = self._auth_conn.authenticate()
        self._set_host(host=upload_host)
        method = 'POST'
        raw = False
        response = self._request(auth_conn=auth_conn, action=action, params=None, data=data, headers=headers, method=method, raw=raw, auth_token=auth_token)
        return response

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False, include_account_id=False):
        params = params or {}
        headers = headers or {}
        auth_conn = self._auth_conn.authenticate()
        self._set_host(host=auth_conn.api_host)
        if not raw and data:
            headers['Content-Type'] = 'application/json'
        if include_account_id:
            if method == 'GET':
                params['accountId'] = auth_conn.account_id
            elif method == 'POST':
                data = data or {}
                data['accountId'] = auth_conn.account_id
        action = API_PATH + action
        if data:
            data = json.dumps(data)
        response = self._request(auth_conn=self._auth_conn, action=action, params=params, data=data, method=method, headers=headers, raw=raw)
        return response

    def _request(self, auth_conn, action, params=None, data=None, headers=None, method='GET', raw=False, auth_token=None):
        params = params or {}
        headers = headers or {}
        if not auth_token:
            auth_token = self._auth_conn.auth_token
        headers['Authorization'] = '%s' % auth_token
        response = super().request(action=action, params=params, data=data, method=method, headers=headers, raw=raw)
        return response

    def _set_host(self, host):
        """
        Dynamically set host which will be used for the following HTTP
        requests.

        NOTE: This is needed because Backblaze uses different hosts for API,
        download and upload requests.
        """
        self.host = host
        self.connection.host = 'https://%s' % host