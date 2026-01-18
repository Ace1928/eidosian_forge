from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyAlgorithmValueValuesEnum(_messages.Enum):
    """Specifies the algorithm (and possibly key size) for the key.

    Values:
      KEY_ALG_UNSPECIFIED: An unspecified key algorithm.
      KEY_ALG_RSA_1024: 1k RSA Key.
      KEY_ALG_RSA_2048: 2k RSA Key.
    """
    KEY_ALG_UNSPECIFIED = 0
    KEY_ALG_RSA_1024 = 1
    KEY_ALG_RSA_2048 = 2