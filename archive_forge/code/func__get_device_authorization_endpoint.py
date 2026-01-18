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
def _get_device_authorization_endpoint(self, session):
    """Get the endpoint for the OAuth 2.0 Device Authorization flow.

        This method will return the correct device authorization endpoint to
        be used.
        If the user has explicitly passed an device_authorization_endpoint to
        the constructor that will be returned. If there is no explicit endpoint
        and a discovery url is provided, it will try to get it from the
        discovery document. If nothing is found, an exception will be raised.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :return: the endpoint to use
        :rtype: string or None if no endpoint is found
        """
    if self.device_authorization_endpoint is not None:
        return self.device_authorization_endpoint
    discovery = self._get_discovery_document(session)
    endpoint = discovery.get('device_authorization_endpoint')
    if endpoint is None:
        raise exceptions.oidc.OidcDeviceAuthorizationEndpointNotFound()
    return endpoint