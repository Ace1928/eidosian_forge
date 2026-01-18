import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
def _bad_oauth_token(self, response, content):
    """Helper method to detect an error caused by a bad OAuth token."""
    return response.status == 401 and (content.startswith(b'Expired token') or content.startswith(b'Invalid token') or content.startswith(b'Unknown access token'))