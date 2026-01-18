import time
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import urlparse, urlencode, basestring
from libcloud.common.base import BaseDriver, RawResponse, JsonResponse, ConnectionUserAndKey
class AzureResourceManagementConnection(ConnectionUserAndKey):
    """
    Represents a single connection to Azure
    """
    conn_class = LibcloudConnection
    driver = AzureBaseDriver
    name = 'Azure AD Auth'
    responseCls = AzureJsonResponse
    rawResponseCls = RawResponse

    def __init__(self, key, secret, secure=True, tenant_id=None, subscription_id=None, cloud_environment=None, **kwargs):
        super().__init__(key, secret, **kwargs)
        if not cloud_environment:
            cloud_environment = 'default'
        if isinstance(cloud_environment, basestring):
            cloud_environment = publicEnvironments[cloud_environment]
        if not isinstance(cloud_environment, dict):
            raise Exception("cloud_environment must be one of '%s' or a dict containing keys 'resourceManagerEndpointUrl', 'activeDirectoryEndpointUrl', 'activeDirectoryResourceId', 'storageEndpointSuffix'" % "', '".join(publicEnvironments.keys()))
        self.host = urlparse.urlparse(cloud_environment['resourceManagerEndpointUrl']).hostname
        self.login_host = urlparse.urlparse(cloud_environment['activeDirectoryEndpointUrl']).hostname
        self.login_resource = cloud_environment['activeDirectoryResourceId']
        self.storage_suffix = cloud_environment['storageEndpointSuffix']
        self.tenant_id = tenant_id
        self.subscription_id = subscription_id

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'application/json'
        headers['Authorization'] = 'Bearer %s' % self.access_token
        return headers

    def encode_data(self, data):
        """Encode data to JSON"""
        return json.dumps(data)

    def get_token_from_credentials(self):
        """
        Log in and get bearer token used to authorize API requests.
        """
        conn = self.conn_class(self.login_host, 443, timeout=self.timeout)
        conn.connect()
        params = urlencode({'grant_type': 'client_credentials', 'client_id': self.user_id, 'client_secret': self.key, 'resource': self.login_resource})
        headers = {'Content-type': 'application/x-www-form-urlencoded'}
        conn.request('POST', '/%s/oauth2/token' % self.tenant_id, params, headers)
        js = AzureAuthJsonResponse(conn.getresponse(), conn)
        self.access_token = js.object['access_token']
        self.expires_on = js.object['expires_on']

    def connect(self, **kwargs):
        self.get_token_from_credentials()
        return super().connect(**kwargs)

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False):
        if time.time() + 300 >= int(self.expires_on):
            self.get_token_from_credentials()
        return super().request(action, params=params, data=data, headers=headers, method=method, raw=raw)