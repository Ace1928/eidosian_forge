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
class OidcClientCredentials(_OidcBase):
    """Implementation for OpenID Connect Client Credentials."""
    grant_type = 'client_credentials'

    def __init__(self, auth_url, identity_provider, protocol, client_id, client_secret, access_token_endpoint=None, discovery_endpoint=None, access_token_type='access_token', **kwargs):
        """The OpenID Client Credentials expects the following.

        :param client_id: Client ID used to authenticate
        :type username: string

        :param client_secret: Client Secret used to authenticate
        :type password: string
        """
        super(OidcClientCredentials, self).__init__(auth_url=auth_url, identity_provider=identity_provider, protocol=protocol, client_id=client_id, client_secret=client_secret, access_token_endpoint=access_token_endpoint, discovery_endpoint=discovery_endpoint, access_token_type=access_token_type, **kwargs)

    def get_payload(self, session):
        """Get an authorization grant for the client credentials grant type.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a python dictionary containing the payload to be exchanged
        :rtype: dict
        """
        payload = {'scope': self.scope}
        return payload