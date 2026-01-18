import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import safe_crypt, test_crypt, to_unicode
from passlib.utils.binary import h64, h64big
from passlib.utils.compat import byte_elem_value, u, uascii_to_str, unicode, suppress_cause
from passlib.crypto.des import des_encrypt_int_block
import passlib.utils.handlers as uh
@classmethod
def _generate_rounds(cls):
    rounds = super(bsdi_crypt, cls)._generate_rounds()
    return rounds | 1