import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def _encode_caveat_v1(condition, root_key, third_party_pub_key, key):
    """Create a JSON-encoded third-party caveat.

    The third_party_pub_key key represents the PublicKey of the third party
    we're encrypting the caveat for; the key is the public/private key pair of
    the party that's adding the caveat.

    @param condition string
    @param root_key bytes
    @param third_party_pub_key (PublicKey)
    @param key (PrivateKey)
    @return a base64 encoded bytes
    """
    plain_data = json.dumps({'RootKey': base64.b64encode(root_key).decode('ascii'), 'Condition': condition})
    box = nacl.public.Box(key.key, third_party_pub_key.key)
    encrypted = box.encrypt(six.b(plain_data))
    nonce = encrypted[0:nacl.public.Box.NONCE_SIZE]
    encrypted = encrypted[nacl.public.Box.NONCE_SIZE:]
    return base64.b64encode(six.b(json.dumps({'ThirdPartyPublicKey': str(third_party_pub_key), 'FirstPartyPublicKey': str(key.public_key), 'Nonce': base64.b64encode(nonce).decode('ascii'), 'Id': base64.b64encode(encrypted).decode('ascii')})))