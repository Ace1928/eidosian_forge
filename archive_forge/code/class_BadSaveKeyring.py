from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class BadSaveKeyring:
    """A keyring that generates errors when saving passwords."""

    def get_password(self, service, username):
        return None

    def set_password(self, service, username, password):
        raise RuntimeError