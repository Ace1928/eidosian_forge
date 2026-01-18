from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeySpecValueValuesEnum(_messages.Enum):
    """Required. The specifications for the key.

    Values:
      KEY_SPEC_UNSPECIFIED: No key specification specified.
      RSA_2048: A 2048 bit RSA key.
      RSA_3072: A 3072 bit RSA key.
      RSA_4096: A 4096 bit RSA key.
    """
    KEY_SPEC_UNSPECIFIED = 0
    RSA_2048 = 1
    RSA_3072 = 2
    RSA_4096 = 3