import abc
import base64
import json
import logging
import os
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery._utils import b64decode
from pymacaroons.serializers import json_serializer
from ._versions import (
from ._error import (
from ._codec import (
from ._keys import PublicKey
from ._third_party import (
def _new_caveat_id(self, base):
    """Return a third party caveat id

        This does not duplicate any third party caveat ids already inside
        macaroon. If base is non-empty, it is used as the id prefix.

        @param base bytes
        @return bytes
        """
    id = bytearray()
    if len(base) > 0:
        id.extend(base)
    else:
        id.append(VERSION_3)
    i = len(self._caveat_data)
    caveats = self._macaroon.caveats
    while True:
        temp = id[:]
        encode_uvarint(i, temp)
        found = False
        for cav in caveats:
            if cav.verification_key_id is not None and cav.caveat_id == temp:
                found = True
                break
        if not found:
            return bytes(temp)
        i += 1