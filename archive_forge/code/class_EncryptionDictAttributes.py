from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class EncryptionDictAttributes:
    """
    Additional encryption dictionary entries for the standard security handler.

    TABLE 3.19, Page 122
    """
    R = '/R'
    O = '/O'
    U = '/U'
    P = '/P'
    ENCRYPT_METADATA = '/EncryptMetadata'