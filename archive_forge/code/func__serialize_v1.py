import binascii
import json
from pymacaroons import utils
def _serialize_v1(self, macaroon):
    """Serialize the macaroon in JSON format v1.

        @param macaroon the macaroon to serialize.
        @return JSON macaroon.
        """
    serialized = {'identifier': utils.convert_to_string(macaroon.identifier), 'signature': macaroon.signature}
    if macaroon.location:
        serialized['location'] = macaroon.location
    if macaroon.caveats:
        serialized['caveats'] = [_caveat_v1_to_dict(caveat) for caveat in macaroon.caveats]
    return json.dumps(serialized)