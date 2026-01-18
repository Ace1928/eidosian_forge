import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
class LaunchpadOAuthAwareHttp(RestfulHttp):
    """Detects expired/invalid OAuth tokens and tries to get a new token."""

    def __init__(self, launchpad, authorization_engine, *args):
        self.launchpad = launchpad
        self.authorization_engine = authorization_engine
        super(LaunchpadOAuthAwareHttp, self).__init__(*args)

    def _bad_oauth_token(self, response, content):
        """Helper method to detect an error caused by a bad OAuth token."""
        return response.status == 401 and (content.startswith(b'Expired token') or content.startswith(b'Invalid token') or content.startswith(b'Unknown access token'))

    def _request(self, *args):
        response, content = super(LaunchpadOAuthAwareHttp, self)._request(*args)
        return self.retry_on_bad_token(response, content, *args)

    def retry_on_bad_token(self, response, content, *args):
        """If the response indicates a bad token, get a new token and retry.

        Otherwise, just return the response.
        """
        if self._bad_oauth_token(response, content) and self.authorization_engine is not None:
            self.launchpad.credentials.access_token = None
            self.authorization_engine(self.launchpad.credentials, self.launchpad.credential_store)
            return self._request(*args)
        return (response, content)