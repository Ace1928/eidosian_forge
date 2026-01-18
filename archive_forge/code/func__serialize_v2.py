import binascii
import json
from pymacaroons import utils
def _serialize_v2(self, macaroon):
    """Serialize the macaroon in JSON format v2.

        @param macaroon the macaroon to serialize.
        @return JSON macaroon in v2 format.
        """
    serialized = {}
    _add_json_binary_field(macaroon.identifier_bytes, serialized, 'i')
    _add_json_binary_field(binascii.unhexlify(macaroon.signature_bytes), serialized, 's')
    if macaroon.location:
        serialized['l'] = macaroon.location
    if macaroon.caveats:
        serialized['c'] = [_caveat_v2_to_dict(caveat) for caveat in macaroon.caveats]
    return json.dumps(serialized)