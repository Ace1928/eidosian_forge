from ..backend import KeyringBackend
from .._compat import properties
from ..errors import NoKeyringError

    Keyring that raises error on every operation.

    >>> kr = Keyring()
    >>> kr.get_password('svc', 'user')
    Traceback (most recent call last):
    ...
    keyring.errors.NoKeyringError: ...No recommended backend...
    