import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentityConnection(ConnectionUserAndKey):
    """
    Base identity connection class which contains common / shared logic.

    Note: This class shouldn't be instantiated directly.
    """
    responseCls = OpenStackAuthResponse
    timeout = None
    auth_version = None

    def __init__(self, auth_url, user_id, key, tenant_name=None, tenant_domain_id='default', domain_name='Default', token_scope=OpenStackIdentityTokenScope.PROJECT, timeout=None, proxy_url=None, parent_conn=None, auth_cache=None):
        super().__init__(user_id=user_id, key=key, url=auth_url, timeout=timeout, proxy_url=proxy_url)
        self.parent_conn = parent_conn
        if parent_conn:
            self.conn_class = parent_conn.conn_class
            self.driver = parent_conn.driver
        else:
            self.driver = None
        self.auth_url = auth_url
        self.tenant_name = tenant_name
        self.domain_name = domain_name
        self.tenant_domain_id = tenant_domain_id
        self.token_scope = token_scope
        self.timeout = timeout
        self.auth_cache = auth_cache
        self.urls = {}
        self.auth_token = None
        self.auth_token_expires = None
        self.auth_user_info = None
        self.auth_user_roles = None

    def authenticated_request(self, action, params=None, data=None, headers=None, method='GET', raw=False):
        """
        Perform an authenticated request against the identity API.
        """
        if not self.auth_token:
            raise ValueError('Need to be authenticated to perform this request')
        headers = headers or {}
        headers[AUTH_TOKEN_HEADER] = self.auth_token
        response = self.request(action=action, params=params, data=data, headers=headers, method=method, raw=raw)
        if response.status == httplib.UNAUTHORIZED:
            self.clear_cached_auth_context()
        return response

    def morph_action_hook(self, action):
        _, _, _, request_path = self._tuple_from_url(self.auth_url)
        if request_path == '':
            return action
        return super().morph_action_hook(action=action)

    def add_default_headers(self, headers):
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json; charset=UTF-8'
        return headers

    def is_token_valid(self):
        """
        Return True if the current auth token is already cached and hasn't
        expired yet.

        :return: ``True`` if the token is still valid, ``False`` otherwise.
        :rtype: ``bool``
        """
        if not self.auth_token:
            return False
        if not self.auth_token_expires:
            return False
        expires = self.auth_token_expires - datetime.timedelta(seconds=AUTH_TOKEN_EXPIRES_GRACE_SECONDS)
        time_tuple_expires = expires.utctimetuple()
        time_tuple_now = datetime.datetime.utcnow().utctimetuple()
        if time_tuple_now < time_tuple_expires:
            return True
        return False

    def authenticate(self, force=False):
        """
        Authenticate against the identity API.

        :param force: Forcefully update the token even if it's already cached
                      and still valid.
        :type force: ``bool``
        """
        raise NotImplementedError('authenticate not implemented')

    def clear_cached_auth_context(self):
        """
        Clear the cached authentication context.

        The context is cleared from fields on this connection and from the
        external cache, if one is configured.
        """
        self.auth_token = None
        self.auth_token_expires = None
        self.auth_user_info = None
        self.auth_user_roles = None
        self.urls = {}
        if self.auth_cache is not None:
            self.auth_cache.clear(self._cache_key)

    def list_supported_versions(self):
        """
        Retrieve a list of all the identity versions which are supported by
        this installation.

        :rtype: ``list`` of :class:`.OpenStackIdentityVersion`
        """
        response = self.request('/', method='GET')
        result = self._to_versions(data=response.object['versions']['values'])
        result = sorted(result, key=lambda x: x.version)
        return result

    def _to_versions(self, data):
        result = []
        for item in data:
            version = self._to_version(data=item)
            result.append(version)
        return result

    def _to_version(self, data):
        try:
            updated = parse_date(data['updated'])
        except Exception:
            updated = None
        try:
            url = data['links'][0]['href']
        except IndexError:
            url = None
        version = OpenStackIdentityVersion(version=data['id'], status=data['status'], updated=updated, url=url)
        return version

    def _is_authentication_needed(self, force=False):
        """
        Determine if the authentication is needed or if the existing token (if
        any exists) is still valid.
        """
        if force:
            return True
        if self.auth_version not in AUTH_VERSIONS_WITH_EXPIRES:
            return True
        if self.is_token_valid():
            return False
        self._load_auth_context_from_cache()
        if self.is_token_valid():
            return False
        return True

    def _to_projects(self, data):
        result = []
        for item in data:
            project = self._to_project(data=item)
            result.append(project)
        return result

    def _to_project(self, data):
        project = OpenStackIdentityProject(id=data['id'], name=data['name'], description=data['description'], enabled=data['enabled'], domain_id=data.get('domain_id', None))
        return project

    @property
    def _cache_key(self):
        """
        The key where this connection's authentication context will be cached.

        :rtype: :class:`OpenStackAuthenticationCacheKey`
        """
        return OpenStackAuthenticationCacheKey(self.auth_url, self.user_id, self.token_scope, self.tenant_name, self.domain_name, self.tenant_domain_id)

    def _cache_auth_context(self, context):
        """
        Store an authentication context in memory and the cache.

        :param context: Authentication context to cache.
        :type key: :class:`.OpenStackAuthenticationContext`
        """
        self.urls = context.urls
        self.auth_token = context.token
        self.auth_token_expires = context.expiration
        self.auth_user_info = context.user
        self.auth_user_roles = context.roles
        if self.auth_cache is not None:
            self.auth_cache.put(self._cache_key, context)

    def _load_auth_context_from_cache(self):
        """
        Fetch an authentication context for this connection from the cache.

        :rtype: :class:`OpenStackAuthenticationContext`
        """
        if self.auth_cache is None:
            return None
        context = self.auth_cache.get(self._cache_key)
        if context is None:
            return None
        self.urls = context.urls
        self.auth_token = context.token
        self.auth_token_expires = context.expiration
        self.auth_user_info = context.user
        self.auth_user_roles = context.roles
        return context