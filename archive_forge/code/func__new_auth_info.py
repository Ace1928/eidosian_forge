from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def _new_auth_info(self, used):
    info = AuthInfo(identity=self._identity, macaroons=[])
    for i, is_used in enumerate(used):
        if is_used:
            info.macaroons.append(self._macaroons[i])
    return info