from base64 import b64encode, b64decode
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_unicode
import passlib.utils.handlers as uh
from passlib.utils.compat import bascii_to_str, iteritems, u,\
from passlib.crypto.digest import pbkdf1
@classmethod
def _norm_variant(cls, variant):
    if isinstance(variant, bytes):
        variant = variant.decode('ascii')
    if isinstance(variant, unicode):
        try:
            variant = cls._variant_aliases[variant]
        except KeyError:
            raise ValueError('invalid fshp variant')
    if not isinstance(variant, int):
        raise TypeError('fshp variant must be int or known alias')
    if variant not in cls._variant_info:
        raise ValueError('invalid fshp variant')
    return variant