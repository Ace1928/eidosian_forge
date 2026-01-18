import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
class OidcAuthorizationCode(_OidcBase):
    """Implementation for OpenID Connect Authorization Code."""
    grant_type = 'authorization_code'

    def __init__(self, auth_url, identity_provider, protocol, client_id, client_secret, access_token_endpoint=None, discovery_endpoint=None, access_token_type='access_token', redirect_uri=None, code=None, **kwargs):
        """The OpenID Authorization Code plugin expects the following.

        :param redirect_uri: OpenID Connect Client Redirect URL
        :type redirect_uri: string

        :param code: OAuth 2.0 Authorization Code
        :type code: string

        """
        super(OidcAuthorizationCode, self).__init__(auth_url=auth_url, identity_provider=identity_provider, protocol=protocol, client_id=client_id, client_secret=client_secret, access_token_endpoint=access_token_endpoint, discovery_endpoint=discovery_endpoint, access_token_type=access_token_type, **kwargs)
        self.redirect_uri = redirect_uri
        self.code = code

    def get_payload(self, session):
        """Get an authorization grant for the "authorization_code" grant type.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a python dictionary containing the payload to be exchanged
        :rtype: dict
        """
        payload = {'redirect_uri': self.redirect_uri, 'code': self.code}
        return payload