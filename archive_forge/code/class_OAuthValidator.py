from keystone.common import provider_api
from keystone import exception
from keystone.oauth1.backends import base
from keystone.oauth1 import core as oauth1
class OAuthValidator(provider_api.ProviderAPIMixin, oauth1.RequestValidator):

    @property
    def enforce_ssl(self):
        return False

    @property
    def safe_characters(self):
        return set('abcdef0123456789')

    def _check_token(self, token):
        return set(token) <= self.safe_characters and len(token) == 32

    def check_client_key(self, client_key):
        return self._check_token(client_key)

    def check_request_token(self, request_token):
        return self._check_token(request_token)

    def check_access_token(self, access_token):
        return self._check_token(access_token)

    def check_nonce(self, nonce):
        return set(nonce) <= self.safe_characters

    def check_verifier(self, verifier):
        return all((i in base.VERIFIER_CHARS for i in verifier)) and len(verifier) == 8

    def get_client_secret(self, client_key, request):
        client = PROVIDERS.oauth_api.get_consumer_with_secret(client_key)
        return client['secret']

    def get_request_token_secret(self, client_key, token, request):
        token_ref = PROVIDERS.oauth_api.get_request_token(token)
        return token_ref['request_secret']

    def get_access_token_secret(self, client_key, token, request):
        access_token = PROVIDERS.oauth_api.get_access_token(token)
        return access_token['access_secret']

    def get_default_realms(self, client_key, request):
        return []

    def get_realms(self, token, request):
        return []

    def get_redirect_uri(self, token, request):
        return 'oob'

    def get_rsa_key(self, client_key, request):
        return ''

    def invalidate_request_token(self, client_key, request_token, request):
        """Invalidate a used request token.

        :param client_key: The client/consumer key.
        :param request_token: The request token string.
        :param request: An oauthlib.common.Request object.
        :returns: None

        Per `Section 2.3`_ of the spec:

        "The server MUST (...) ensure that the temporary
        credentials have not expired or been used before."

        .. _`Section 2.3`: https://tools.ietf.org/html/rfc5849#section-2.3

        This method should ensure that provided token won't validate anymore.
        It can be simply removing RequestToken from storage or setting
        specific flag that makes it invalid (note that such flag should be
        also validated during request token validation).

        This method is used by

        * AccessTokenEndpoint
        """
        pass

    def validate_client_key(self, client_key, request):
        try:
            return PROVIDERS.oauth_api.get_consumer(client_key) is not None
        except exception.NotFound:
            return False

    def validate_request_token(self, client_key, token, request):
        try:
            req_token = PROVIDERS.oauth_api.get_request_token(token)
            if req_token:
                return req_token['consumer_id'] == client_key
            else:
                return False
        except exception.NotFound:
            return False

    def validate_access_token(self, client_key, token, request):
        try:
            return PROVIDERS.oauth_api.get_access_token(token) is not None
        except exception.NotFound:
            return False

    def validate_timestamp_and_nonce(self, client_key, timestamp, nonce, request, request_token=None, access_token=None):
        return True

    def validate_redirect_uri(self, client_key, redirect_uri, request):
        return True

    def validate_requested_realms(self, client_key, realms, request):
        return True

    def validate_realms(self, client_key, token, request, uri=None, realms=None):
        return True

    def validate_verifier(self, client_key, token, verifier, request):
        try:
            req_token = PROVIDERS.oauth_api.get_request_token(token)
            return req_token['verifier'] == verifier
        except exception.NotFound:
            return False

    def verify_request_token(self, token, request):
        return isinstance(token, str)

    def verify_realms(self, token, realms, request):
        return True

    def save_access_token(self, token, request):
        pass

    def save_request_token(self, token, request):
        pass

    def save_verifier(self, token, verifier, request):
        """Associate an authorization verifier with a request token.

        :param token: A request token string.
        :param verifier: A dictionary containing the oauth_verifier and
                         oauth_token
        :param request: An oauthlib.common.Request object.

        We need to associate verifiers with tokens for validation during the
        access token request.

        Note that unlike save_x_token token here is the ``oauth_token`` token
        string from the request token saved previously.

        This method is used by

        * AuthorizationEndpoint
        """
        pass