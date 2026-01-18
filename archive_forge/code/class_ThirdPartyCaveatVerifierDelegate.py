from __future__ import unicode_literals
import binascii
from nacl.secret import SecretBox
from pymacaroons import Caveat
from pymacaroons.utils import (
from pymacaroons.exceptions import MacaroonUnmetCaveatException
from .base_third_party import (
class ThirdPartyCaveatVerifierDelegate(BaseThirdPartyCaveatVerifierDelegate):

    def __init__(self, *args, **kwargs):
        super(ThirdPartyCaveatVerifierDelegate, self).__init__(*args, **kwargs)

    def verify_third_party_caveat(self, verifier, caveat, root, macaroon, discharge_macaroons, signature):
        caveat_macaroon = self._caveat_macaroon(caveat, discharge_macaroons)
        caveat_key = self._extract_caveat_key(signature, caveat)
        caveat_met = verifier.verify_discharge(root, caveat_macaroon, caveat_key, discharge_macaroons=discharge_macaroons)
        return caveat_met

    def update_signature(self, signature, caveat):
        return binascii.unhexlify(sign_third_party_caveat(signature, caveat._verification_key_id, caveat._caveat_id))

    def _caveat_macaroon(self, caveat, discharge_macaroons):
        caveat_macaroon = next((m for m in discharge_macaroons if m.identifier_bytes == caveat.caveat_id_bytes), None)
        if not caveat_macaroon:
            raise MacaroonUnmetCaveatException('Caveat not met. No discharge macaroon found for identifier: {}'.format(caveat.caveat_id_bytes))
        return caveat_macaroon

    def _extract_caveat_key(self, signature, caveat):
        key = truncate_or_pad(signature)
        box = SecretBox(key=key)
        decrypted = box.decrypt(caveat._verification_key_id)
        return decrypted