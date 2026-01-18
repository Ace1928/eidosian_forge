import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve

        :type privateSigningKey: ECPrivateKey
        :type  message: bytearray
        