import base64
import hashlib
import itertools
import os
import google
from ._checker import (Op, LOGIN_OP)
from ._store import MemoryKeyStore
from ._error import VerificationError
from ._versions import (
from ._macaroon import (
import macaroonbakery.checkers as checkers
import six
from macaroonbakery._utils import (
from ._internal import id_pb2
from pymacaroons import MACAROON_V2, Verifier
 This method makes the oven satisfy the MacaroonOpStore protocol
        required by the Checker class.

        For macaroons minted with previous bakery versions, it always
        returns a single LoginOp operation.

        :param macaroons:
        :return:
        