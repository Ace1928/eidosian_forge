import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
Decode a variable-length integer.

    Reads a sequence of unsigned integer byte and decodes them into an integer
    in variable-length format and returns it and the length read.
    