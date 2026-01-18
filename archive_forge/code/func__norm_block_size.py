from __future__ import with_statement, absolute_import
import logging; log = logging.getLogger(__name__)
from passlib.crypto import scrypt as _scrypt
from passlib.utils import h64, to_bytes
from passlib.utils.binary import h64, b64s_decode, b64s_encode
from passlib.utils.compat import u, bascii_to_str, suppress_cause
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
@classmethod
def _norm_block_size(cls, block_size, relaxed=False):
    return uh.norm_integer(cls, block_size, min=1, param='block_size', relaxed=relaxed)