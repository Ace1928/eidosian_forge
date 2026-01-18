from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
class KeyringCredentialStore(CredentialStore):
    """Store credentials in the GNOME keyring or KDE wallet.

    This is a good solution for desktop applications and interactive
    scripts. It doesn't work for non-interactive scripts, or for
    integrating third-party websites into Launchpad.
    """
    B64MARKER = b'<B64>'

    def __init__(self, credential_save_failed=None, fallback=False):
        super(KeyringCredentialStore, self).__init__(credential_save_failed)
        self._fallback = None
        if fallback:
            self._fallback = MemoryCredentialStore(credential_save_failed)

    @staticmethod
    def _ensure_keyring_imported():
        """Ensure the keyring module is imported (postponing side effects).

        The keyring module initializes the environment-dependent backend at
        import time (nasty).  We want to avoid that initialization because it
        may do things like prompt the user to unlock their password store
        (e.g., KWallet).
        """
        if 'keyring' not in globals():
            global keyring
            import keyring
        if 'NoKeyringError' not in globals():
            global NoKeyringError
            try:
                from keyring.errors import NoKeyringError
            except ImportError:
                NoKeyringError = RuntimeError

    def do_save(self, credentials, unique_key):
        """Store newly-authorized credentials in the keyring."""
        self._ensure_keyring_imported()
        serialized = credentials.serialize()
        serialized = self.B64MARKER + b64encode(serialized)
        try:
            keyring.set_password('launchpadlib', unique_key, serialized.decode('utf-8'))
        except NoKeyringError as e:
            if NoKeyringError == RuntimeError and 'No recommended backend was available' not in str(e):
                raise
            if self._fallback:
                self._fallback.save(credentials, unique_key)
            else:
                raise

    def do_load(self, unique_key):
        """Retrieve credentials from the keyring."""
        self._ensure_keyring_imported()
        try:
            credential_string = keyring.get_password('launchpadlib', unique_key)
        except NoKeyringError as e:
            if NoKeyringError == RuntimeError and 'No recommended backend was available' not in str(e):
                raise
            if self._fallback:
                return self._fallback.load(unique_key)
            else:
                raise
        if credential_string is not None:
            if isinstance(credential_string, unicode_type):
                credential_string = credential_string.encode('utf8')
            if credential_string.startswith(self.B64MARKER):
                try:
                    credential_string = b64decode(credential_string[len(self.B64MARKER):])
                except TypeError:
                    return None
            try:
                credentials = Credentials.from_string(credential_string)
                return credentials
            except Exception:
                return None
        return None