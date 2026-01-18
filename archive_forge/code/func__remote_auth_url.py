from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
from keystoneauth1 import plugin
@classmethod
def _remote_auth_url(cls, auth_url):
    """Return auth_url of the remote Keystone Service Provider.

        Remote cloud's auth_url is an endpoint for getting federated unscoped
        token, typically that would be
        ``https://remote.example.com:5000/v3/OS-FEDERATION/identity_providers/
        <idp>/protocols/<protocol_id>/auth``. However we need to generate a
        real auth_url, used for token scoping.  This function assumes there are
        static values today in the remote auth_url stored in the Service
        Provider attribute and those can be used as a delimiter. If the
        sp_auth_url doesn't comply with standard federation auth url the
        function will simply return whole string.

        :param auth_url: auth_url of the remote cloud
        :type auth_url: str

        :returns: auth_url of remote cloud where a token can be validated or
                  scoped.
        :rtype: str

        """
    PATTERN = '/OS-FEDERATION/'
    idx = auth_url.index(PATTERN) if PATTERN in auth_url else len(auth_url)
    return auth_url[:idx]