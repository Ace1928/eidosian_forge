from __future__ import unicode_literals
import binascii
from nacl.secret import SecretBox
from pymacaroons import Caveat
from pymacaroons.utils import (
from pymacaroons.exceptions import MacaroonUnmetCaveatException
from .base_third_party import (
def _extract_caveat_key(self, signature, caveat):
    key = truncate_or_pad(signature)
    box = SecretBox(key=key)
    decrypted = box.decrypt(caveat._verification_key_id)
    return decrypted