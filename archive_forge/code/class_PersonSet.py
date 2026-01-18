import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
class PersonSet(CollectionWithKeyBasedLookup):
    """A custom subclass capable of person lookup by username."""

    def _get_url_from_id(self, key):
        """Transform a username into the URL to a person resource."""
        return str(self._root._root_uri.ensureSlash()) + '~' + str(key)
    collection_of = 'team'