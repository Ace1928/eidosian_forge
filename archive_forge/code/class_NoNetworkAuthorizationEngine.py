from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class NoNetworkAuthorizationEngine(RequestTokenAuthorizationEngine):
    """An authorization engine that doesn't open a web browser.

    You can use this to test the creation of Launchpad objects and the
    storing of credentials. You can't use it to interact with the web
    service, since it only pretends to authorize its OAuth request tokens.
    """
    ACCESS_TOKEN_KEY = 'access_key:84'

    def __init__(self, *args, **kwargs):
        super(NoNetworkAuthorizationEngine, self).__init__(*args, **kwargs)
        self.request_tokens_obtained = 0
        self.access_tokens_obtained = 0

    def get_request_token(self, credentials):
        """Pretend to get a request token from the server.

        We do this by simply returning a static token ID.
        """
        self.request_tokens_obtained += 1
        return 'request_token:42'

    def make_end_user_authorize_token(self, credentials, request_token):
        """Pretend to exchange a request token for an access token.

        We do this by simply setting the access_token property.
        """
        credentials.access_token = AccessToken(self.ACCESS_TOKEN_KEY, 'access_secret:168')
        self.access_tokens_obtained += 1