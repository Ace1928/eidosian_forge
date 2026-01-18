from __future__ import unicode_literals
import binascii
from nacl.secret import SecretBox
from pymacaroons import Caveat
from pymacaroons.utils import (
from pymacaroons.exceptions import MacaroonUnmetCaveatException
from .base_third_party import (
def _caveat_macaroon(self, caveat, discharge_macaroons):
    caveat_macaroon = next((m for m in discharge_macaroons if m.identifier_bytes == caveat.caveat_id_bytes), None)
    if not caveat_macaroon:
        raise MacaroonUnmetCaveatException('Caveat not met. No discharge macaroon found for identifier: {}'.format(caveat.caveat_id_bytes))
    return caveat_macaroon