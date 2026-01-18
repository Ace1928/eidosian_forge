import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
@classmethod
def _is_sudo(cls):
    return {'SUDO_USER', 'SUDO_UID', 'SUDO_GID'} & set(os.environ.keys())