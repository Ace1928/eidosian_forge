import binascii
import json
from pymacaroons import utils
def _deserialize_v1(self, deserialized):
    """Deserialize a JSON macaroon in v1 format.

        @param serialized the macaroon in v1 JSON format.
        @return the macaroon object.
        """
    from pymacaroons.macaroon import Macaroon, MACAROON_V1
    from pymacaroons.caveat import Caveat
    caveats = []
    for c in deserialized.get('caveats', []):
        caveat = Caveat(caveat_id=c['cid'], verification_key_id=utils.raw_b64decode(c['vid']) if c.get('vid') else None, location=c['cl'] if c.get('cl') else None, version=MACAROON_V1)
        caveats.append(caveat)
    return Macaroon(location=deserialized.get('location'), identifier=deserialized['identifier'], caveats=caveats, signature=deserialized['signature'], version=MACAROON_V1)