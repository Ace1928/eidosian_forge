import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
class CryptoError(Exception):
    """Base class for all exceptions in this module."""