import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def _needs_reauthenticate(self):
    """Return if the existing token needs to be re-authenticated.

        The token should be refreshed if it is about to expire.

        :returns: True if the plugin should fetch a new token. False otherwise.
        """
    if not self.auth_ref:
        return True
    if not self.reauthenticate:
        return False
    if self.auth_ref.will_expire_soon(self.MIN_TOKEN_LIFE_SECONDS):
        return True
    return False