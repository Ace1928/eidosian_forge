import sys
import os
import contextlib
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError
from ..errors import PasswordSetError, InitError, KeyringLocked
from .._compat import properties
class DBusKeyringKWallet4(DBusKeyring):
    """
    KDE KWallet 4 via D-Bus
    """
    bus_name = 'org.kde.kwalletd'
    object_path = '/modules/kwalletd'

    @properties.classproperty
    def priority(cls):
        return super().priority - 1