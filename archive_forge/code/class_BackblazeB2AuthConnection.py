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
class BackblazeB2AuthConnection(ConnectionUserAndKey):
    host = AUTH_API_HOST
    secure = True
    responseCls = BackblazeB2Response

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.account_id = None
        self.api_url = None
        self.api_host = None
        self.download_url = None
        self.download_host = None
        self.auth_token = None

    def authenticate(self, force=False):
        """
        :param force: Force authentication if if we have already obtained the
                      token.
        :type force: ``bool``
        """
        if not self._is_authentication_needed(force=force):
            return self
        headers = {}
        action = 'b2_authorize_account'
        auth_b64 = base64.b64encode(b('{}:{}'.format(self.user_id, self.key)))
        headers['Authorization'] = 'Basic %s' % auth_b64.decode('utf-8')
        action = API_PATH + 'b2_authorize_account'
        resp = self.request(action=action, headers=headers, method='GET')
        if resp.status == httplib.OK:
            self._parse_and_set_auth_info(data=resp.object)
        else:
            raise Exception('Failed to authenticate: %s' % str(resp.object))
        return self

    def _parse_and_set_auth_info(self, data):
        result = {}
        self.account_id = data['accountId']
        self.api_url = data['apiUrl']
        self.download_url = data['downloadUrl']
        self.auth_token = data['authorizationToken']
        parsed_api_url = urlparse.urlparse(self.api_url)
        self.api_host = parsed_api_url.netloc
        parsed_download_url = urlparse.urlparse(self.download_url)
        self.download_host = parsed_download_url.netloc
        return result

    def _is_authentication_needed(self, force=False):
        if not self.auth_token or force:
            return True
        return False