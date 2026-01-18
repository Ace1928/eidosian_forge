import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, \
from passlib.utils.binary import h64
from passlib.utils.compat import byte_elem_value, u, \
import passlib.utils.handlers as uh
def _parse_rounds(self, rounds):
    return self._norm_rounds(rounds, relaxed=self.checksum is None)