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
def _get_access_token(self, session, payload):
    """Poll token endpoint for an access token.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :param payload: a dict containing various OpenID Connect values,
                for example::
                {'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                 'device_code': self.device_code}
        :type payload: dict
        """
    print(f'\nTo authenticate please go to: {self.verification_uri_complete}')
    client_auth = (self.client_id, self.client_secret)
    access_token_endpoint = self._get_access_token_endpoint(session)
    encoded_payload = urlparse.urlencode(payload)
    while time.time() < self.timeout:
        try:
            op_response = session.post(access_token_endpoint, requests_auth=client_auth, data=encoded_payload, headers=self.HEADER_X_FORM, authenticated=False)
        except exceptions.http.BadRequest as exc:
            error = exc.response.json().get('error')
            if error != 'authorization_pending':
                raise
            time.sleep(self.interval)
            continue
        break
    else:
        if error == 'authorization_pending':
            raise exceptions.oidc.OidcDeviceAuthorizationTimeOut()
    access_token = op_response.json()[self.access_token_type]
    return access_token