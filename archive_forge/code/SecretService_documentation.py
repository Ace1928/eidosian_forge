from contextlib import closing
import logging
from .. import backend
from .._compat import properties
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import (
Gets the first username and password for a service.
        Returns a Credential instance

        The username can be omitted, but if there is one, it will use get_password
        and return a SimpleCredential containing  the username and password
        Otherwise, it will return the first username and password combo that it finds.
        