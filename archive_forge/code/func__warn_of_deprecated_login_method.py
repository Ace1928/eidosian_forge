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
def _warn_of_deprecated_login_method(cls, name):
    warnings.warn('The Launchpad.%s() method is deprecated. You should use Launchpad.login_anonymous() for anonymous access and Launchpad.login_with() for all other purposes.' % name, DeprecationWarning)