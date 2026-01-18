import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_2_0_Connection_VOMS(OpenStackIdentityConnection, CertificateConnection):
    """
    Connection class for Keystone API v2.0. with VOMS proxy support
    In this case the key parameter will be the path of the VOMS proxy file.
    """
    responseCls = OpenStackAuthResponse
    name = 'OpenStack Identity API v2.0 VOMS support'
    auth_version = '2.0'

    def __init__(self, auth_url, user_id, key, tenant_name=None, tenant_domain_id='default', domain_name='Default', token_scope=OpenStackIdentityTokenScope.PROJECT, timeout=None, proxy_url=None, parent_conn=None, auth_cache=None):
        CertificateConnection.__init__(self, cert_file=key, url=auth_url, proxy_url=proxy_url, timeout=timeout)
        self.parent_conn = parent_conn
        if parent_conn:
            self.conn_class = parent_conn.conn_class
            self.driver = parent_conn.driver
        else:
            self.driver = None
        self.auth_url = auth_url
        self.tenant_name = tenant_name
        self.tenant_domain_id = tenant_domain_id
        self.domain_name = domain_name
        self.token_scope = token_scope
        self.timeout = timeout
        self.proxy_url = proxy_url
        self.auth_cache = auth_cache
        self.urls = {}
        self.auth_token = None
        self.auth_token_expires = None
        self.auth_user_info = None

    def authenticate(self, force=False):
        if not self._is_authentication_needed(force=force):
            return self
        tenant = self.tenant_name
        if not tenant:
            token = self._get_unscoped_token()
            tenant = self._get_tenant_name(token)
        data = {'auth': {'voms': True, 'tenantName': tenant}}
        reqbody = json.dumps(data)
        return self._authenticate_2_0_with_body(reqbody)

    def _get_unscoped_token(self):
        """
        Get unscoped token from VOMS proxy
        """
        data = {'auth': {'voms': True}}
        reqbody = json.dumps(data)
        response = self.request('/v2.0/tokens', data=reqbody, headers={'Content-Type': 'application/json'}, method='POST')
        if response.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif response.status in [httplib.OK, httplib.CREATED]:
            try:
                body = json.loads(response.body)
                return body['access']['token']['id']
            except Exception as e:
                raise MalformedResponseError('Failed to parse JSON', e)
        else:
            raise MalformedResponseError('Malformed response', driver=self.driver, body=response.body)

    def _get_tenant_name(self, token):
        """
        Get the first available tenant name (usually there are only one)
        """
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json', AUTH_TOKEN_HEADER: token}
        response = self.request('/v2.0/tenants', headers=headers, method='GET')
        if response.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif response.status in [httplib.OK, httplib.CREATED]:
            try:
                body = json.loads(response.body)
                return body['tenants'][0]['name']
            except Exception as e:
                raise MalformedResponseError('Failed to parse JSON', e)
        else:
            raise MalformedResponseError('Malformed response', driver=self.driver, body=response.body)

    def _authenticate_2_0_with_body(self, reqbody):
        resp = self.request('/v2.0/tokens', data=reqbody, headers={'Content-Type': 'application/json'}, method='POST')
        if resp.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif resp.status not in [httplib.OK, httplib.NON_AUTHORITATIVE_INFORMATION]:
            body = 'code: {} body: {}'.format(resp.status, resp.body)
            raise MalformedResponseError('Malformed response', body=body, driver=self.driver)
        else:
            body = resp.object
            try:
                access = body['access']
                expires = access['token']['expires']
                self._cache_auth_context(OpenStackAuthenticationContext(access['token']['id'], expiration=parse_date(expires), urls=access['serviceCatalog'], user=access.get('user', {})))
            except KeyError as e:
                raise MalformedResponseError('Auth JSON response is                                              missing required elements', e)
        return self