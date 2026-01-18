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
def _get_keystone_token(self, session, access_token):
    """Exchange an access token for a keystone token.

        By Sending the access token in an `Authorization: Bearer` header, to
        an OpenID Connect protected endpoint (Federated Token URL). The
        OpenID Connect server will use the access token to look up information
        about the authenticated user (this technique is called instrospection).
        The output of the instrospection will be an OpenID Connect Claim, that
        will be used against the mapping engine. Should the mapping engine
        succeed, a Keystone token will be presented to the user.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :param access_token: The OpenID Connect access token.
        :type access_token: str
        """
    headers = {'Authorization': 'Bearer ' + access_token}
    auth_response = session.post(self.federated_token_url, headers=headers, authenticated=False)
    return auth_response