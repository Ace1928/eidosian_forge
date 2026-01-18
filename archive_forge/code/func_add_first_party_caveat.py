from __future__ import unicode_literals
import binascii
from pymacaroons import Caveat
from pymacaroons.utils import (
from .base_first_party import (
def add_first_party_caveat(self, macaroon, predicate, **kwargs):
    predicate = convert_to_bytes(predicate)
    predicate.decode('utf-8')
    caveat = Caveat(caveat_id=predicate, version=macaroon.version)
    macaroon.caveats.append(caveat)
    encode_key = binascii.unhexlify(macaroon.signature_bytes)
    macaroon.signature = sign_first_party_caveat(encode_key, predicate)
    return macaroon