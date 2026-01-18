from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
class BearerToken(TokenBase):
    __slots__ = ('request_validator', 'token_generator', 'refresh_token_generator', 'expires_in')

    def __init__(self, request_validator=None, token_generator=None, expires_in=None, refresh_token_generator=None):
        self.request_validator = request_validator
        self.token_generator = token_generator or random_token_generator
        self.refresh_token_generator = refresh_token_generator or self.token_generator
        self.expires_in = expires_in or 3600

    def create_token(self, request, refresh_token=False, **kwargs):
        """
        Create a BearerToken, by default without refresh token.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param refresh_token:
        """
        if 'save_token' in kwargs:
            warnings.warn('`save_token` has been deprecated, it was not called internally.If you do, call `request_validator.save_token()` instead.', DeprecationWarning)
        if callable(self.expires_in):
            expires_in = self.expires_in(request)
        else:
            expires_in = self.expires_in
        request.expires_in = expires_in
        token = {'access_token': self.token_generator(request), 'expires_in': expires_in, 'token_type': 'Bearer'}
        if request.scopes is not None:
            token['scope'] = ' '.join(request.scopes)
        if refresh_token:
            if request.refresh_token and (not self.request_validator.rotate_refresh_token(request)):
                token['refresh_token'] = request.refresh_token
            else:
                token['refresh_token'] = self.refresh_token_generator(request)
        token.update(request.extra_credentials or {})
        return OAuth2Token(token)

    def validate_request(self, request):
        """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
        token = get_token_from_header(request)
        return self.request_validator.validate_bearer_token(token, request.scopes, request)

    def estimate_type(self, request):
        """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
        if request.headers.get('Authorization', '').split(' ')[0] == 'Bearer':
            return 9
        elif request.access_token is not None:
            return 5
        else:
            return 0