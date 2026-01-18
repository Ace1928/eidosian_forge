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
def _assert_login_argument_consistency(cls, argument_name, argument_value, object_value, object_name='authorization engine'):
    """Helper to find conflicting values passed into the login methods.

        Many of the arguments to login_with are used to build other
        objects--the authorization engine or the credential store. If
        these objects are provided directly, many of the arguments
        become redundant. We'll allow redundant arguments through, but
        if a argument *conflicts* with the corresponding value in the
        provided object, we raise an error.
        """
    inconsistent_value_message = "Inconsistent values given for %s: (%r passed in, versus %r in %s). You don't need to pass in %s if you pass in %s, so just omit that argument."
    if argument_value is not None and argument_value != object_value:
        raise ValueError(inconsistent_value_message % (argument_name, argument_value, object_value, object_name, argument_name, object_name))