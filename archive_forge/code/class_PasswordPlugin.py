import datetime
import json
import time
from urllib.parse import urljoin
from keystoneauth1 import discover
from keystoneauth1 import plugin
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.identity import base
class PasswordPlugin(base.BaseIdentityPlugin):
    """A plugin for authenticating with a username and password.

    Subclassing from BaseIdentityPlugin gets us a few niceties, like handling
    token invalidation and locking during authentication.

    :param string auth_url: Identity v1 endpoint for authorization.
    :param string username: Username for authentication.
    :param string password: Password for authentication.
    :param string project_name: Swift account to use after authentication.
                                We use 'project_name' to be consistent with
                                other auth plugins.
    :param string reauthenticate: Whether to allow re-authentication.
    """
    access_class = AccessInfoV1

    def __init__(self, auth_url, username, password, project_name=None, reauthenticate=True):
        super(PasswordPlugin, self).__init__(auth_url=auth_url, reauthenticate=reauthenticate)
        self.user = username
        self.key = password
        self.account = project_name

    def get_auth_ref(self, session, **kwargs):
        """Obtain a token from a v1 endpoint.

        This function should not be called independently and is expected to be
        invoked via the do_authenticate function.

        This function will be invoked if the AcessInfo object cached by the
        plugin is not valid. Thus plugins should always fetch a new AccessInfo
        when invoked. If you are looking to just retrieve the current auth
        data then you should use get_access.

        :param session: A session object that can be used for communication.

        :returns: Token access information.
        """
        headers = {'X-Auth-User': self.user, 'X-Auth-Key': self.key}
        resp = session.get(self.auth_url, headers=headers, authenticated=False, log=False)
        if resp.status_code // 100 != 2:
            raise exceptions.InvalidResponse(response=resp)
        if 'X-Storage-Url' not in resp.headers:
            raise exceptions.InvalidResponse(response=resp)
        if 'X-Auth-Token' not in resp.headers and 'X-Storage-Token' not in resp.headers:
            raise exceptions.InvalidResponse(response=resp)
        token = resp.headers.get('X-Storage-Token', resp.headers.get('X-Auth-Token'))
        return AccessInfoV1(auth_url=self.auth_url, storage_url=resp.headers['X-Storage-Url'], account=self.account, username=self.user, auth_token=token, token_life=resp.headers.get('X-Auth-Token-Expires'))

    def get_cache_id_elements(self):
        """Get the elements for this auth plugin that make it unique."""
        return {'auth_url': self.auth_url, 'user': self.user, 'key': self.key, 'account': self.account}

    def get_endpoint(self, session, interface='public', **kwargs):
        """Return an endpoint for the client."""
        if interface is plugin.AUTH_INTERFACE:
            return self.auth_url
        else:
            return self.get_access(session).service_catalog.url_for(interface=interface, **kwargs)

    def get_auth_state(self):
        """Retrieve the current authentication state for the plugin.

        :returns: raw python data (which can be JSON serialized) that can be
                  moved into another plugin (of the same type) to have the
                  same authenticated state.
        """
        if self.auth_ref:
            return self.auth_ref.get_state()

    def set_auth_state(self, data):
        """Install existing authentication state for a plugin.

        Take the output of get_auth_state and install that authentication state
        into the current authentication plugin.
        """
        if data:
            self.auth_ref = self.access_class.from_state(data)
        else:
            self.auth_ref = None

    def get_sp_auth_url(self, *args, **kwargs):
        raise NotImplementedError()

    def get_sp_url(self, *args, **kwargs):
        raise NotImplementedError()

    def get_discovery(self, *args, **kwargs):
        raise NotImplementedError()