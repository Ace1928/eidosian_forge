from .err import OperationalError
from functools import partial
import hashlib
def _init_nacl():
    global _nacl_bindings
    try:
        from nacl import bindings
        _nacl_bindings = bindings
    except ImportError:
        raise RuntimeError("'pynacl' package is required for ed25519_password auth method")