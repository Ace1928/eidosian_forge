import logging; log = logging.getLogger(__name__)
from itertools import chain
from passlib import hash
from passlib.context import LazyCryptContext
from passlib.utils import sys_bits
def _create_phpass_policy(**kwds):
    """helper to choose default alg based on bcrypt availability"""
    kwds['default'] = 'bcrypt' if hash.bcrypt.has_backend() else 'phpass'
    return kwds