import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def _verify_caveats(self, root, macaroon, discharge_macaroons, signature):
    for caveat in macaroon.caveats:
        if self._caveat_met(root, caveat, macaroon, discharge_macaroons, signature):
            signature = self._update_signature(caveat, signature)
    return signature