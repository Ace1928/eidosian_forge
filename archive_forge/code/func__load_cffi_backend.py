from __future__ import absolute_import
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.utils import to_bytes
from passlib.utils.compat import PYPY
def _load_cffi_backend():
    """
    Try to import the ctypes-based scrypt hash function provided by the
    ``scrypt <https://pypi.python.org/pypi/scrypt/>``_ package.
    """
    try:
        from scrypt import hash
        return hash
    except ImportError:
        pass
    try:
        import scrypt
    except ImportError as err:
        if 'scrypt' not in str(err):
            warn("'scrypt' package failed to import correctly (possible installation issue?)", exc.PasslibWarning)
    else:
        warn("'scrypt' package is too old (lacks ``hash()`` method)", exc.PasslibWarning)
    return None