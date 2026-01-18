import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def _load_keyring_class(keyring_name: str) -> typing.Type[backend.KeyringBackend]:
    """
    Load the keyring class indicated by name.

    These popular names are tested to ensure their presence.

    >>> popular_names = [
    ...      'keyring.backends.Windows.WinVaultKeyring',
    ...      'keyring.backends.macOS.Keyring',
    ...      'keyring.backends.kwallet.DBusKeyring',
    ...      'keyring.backends.SecretService.Keyring',
    ...  ]
    >>> list(map(_load_keyring_class, popular_names))
    [...]
    """
    module_name, sep, class_name = keyring_name.rpartition('.')
    __import__(module_name)
    module = sys.modules[module_name]
    return getattr(module, class_name)