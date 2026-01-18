from __future__ import absolute_import
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.utils import to_bytes
from passlib.utils.compat import PYPY
def _load_builtin_backend():
    """
    Load pure-python scrypt implementation built into passlib.
    """
    slowdown = 10 if PYPY else 100
    warn("Using builtin scrypt backend, which is %dx slower than is required for adequate security. Installing scrypt support (via 'pip install scrypt') is strongly recommended" % slowdown, exc.PasslibSecurityWarning)
    from ._builtin import ScryptEngine
    return ScryptEngine.execute