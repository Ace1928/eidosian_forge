import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class CredentialStore:
    """An abstract class to implement storage for credentials"""

    def decode_password(self, credentials):
        """Returns a clear text password for the provided credentials."""
        raise NotImplementedError(self.decode_password)

    def get_credentials(self, scheme, host, port=None, user=None, path=None, realm=None):
        """Return the matching credentials from this credential store.

        This method is only called on fallback credential stores.
        """
        raise NotImplementedError(self.get_credentials)