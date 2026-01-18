import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
class _JWTAccessCredentials(ServiceAccountCredentials):
    """Self signed JWT credentials.

    Makes an assertion to server using a self signed JWT from service account
    credentials.  These credentials do NOT use OAuth 2.0 and instead
    authenticate directly.
    """
    _MAX_TOKEN_LIFETIME_SECS = 3600
    'Max lifetime of the token (one hour, in seconds).'

    def __init__(self, service_account_email, signer, scopes=None, private_key_id=None, client_id=None, user_agent=None, token_uri=oauth2client.GOOGLE_TOKEN_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI, additional_claims=None):
        if additional_claims is None:
            additional_claims = {}
        super(_JWTAccessCredentials, self).__init__(service_account_email, signer, private_key_id=private_key_id, client_id=client_id, user_agent=user_agent, token_uri=token_uri, revoke_uri=revoke_uri, **additional_claims)

    def authorize(self, http):
        """Authorize an httplib2.Http instance with a JWT assertion.

        Unless specified, the 'aud' of the assertion will be the base
        uri of the request.

        Args:
            http: An instance of ``httplib2.Http`` or something that acts
                  like it.
        Returns:
            A modified instance of http that was passed in.
        Example::
            h = httplib2.Http()
            h = credentials.authorize(h)
        """
        transport.wrap_http_for_jwt_access(self, http)
        return http

    def get_access_token(self, http=None, additional_claims=None):
        """Create a signed jwt.

        Args:
            http: unused
            additional_claims: dict, additional claims to add to
                the payload of the JWT.
        Returns:
            An AccessTokenInfo with the signed jwt
        """
        if additional_claims is None:
            if self.access_token is None or self.access_token_expired:
                self.refresh(None)
            return client.AccessTokenInfo(access_token=self.access_token, expires_in=self._expires_in())
        else:
            token, unused_expiry = self._create_token(additional_claims)
            return client.AccessTokenInfo(access_token=token, expires_in=self._MAX_TOKEN_LIFETIME_SECS)

    def revoke(self, http):
        """Cannot revoke JWTAccessCredentials tokens."""
        pass

    def create_scoped_required(self):
        return True

    def create_scoped(self, scopes, token_uri=oauth2client.GOOGLE_TOKEN_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI):
        result = ServiceAccountCredentials(self._service_account_email, self._signer, scopes=scopes, private_key_id=self._private_key_id, client_id=self.client_id, user_agent=self._user_agent, token_uri=token_uri, revoke_uri=revoke_uri, **self._kwargs)
        if self._private_key_pkcs8_pem is not None:
            result._private_key_pkcs8_pem = self._private_key_pkcs8_pem
        if self._private_key_pkcs12 is not None:
            result._private_key_pkcs12 = self._private_key_pkcs12
        if self._private_key_password is not None:
            result._private_key_password = self._private_key_password
        return result

    def refresh(self, http):
        """Refreshes the access_token.

        The HTTP object is unused since no request needs to be made to
        get a new token, it can just be generated locally.

        Args:
            http: unused HTTP object
        """
        self._refresh(None)

    def _refresh(self, http):
        """Refreshes the access_token.

        Args:
            http: unused HTTP object
        """
        self.access_token, self.token_expiry = self._create_token()

    def _create_token(self, additional_claims=None):
        now = client._UTCNOW()
        lifetime = datetime.timedelta(seconds=self._MAX_TOKEN_LIFETIME_SECS)
        expiry = now + lifetime
        payload = {'iat': _datetime_to_secs(now), 'exp': _datetime_to_secs(expiry), 'iss': self._service_account_email, 'sub': self._service_account_email}
        payload.update(self._kwargs)
        if additional_claims is not None:
            payload.update(additional_claims)
        jwt = crypt.make_signed_jwt(self._signer, payload, key_id=self._private_key_id)
        return (jwt.decode('ascii'), expiry)