from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class PBES(utils.Enum):
    PBESv1SHA1And3KeyTripleDESCBC = 'PBESv1 using SHA1 and 3-Key TripleDES'
    PBESv2SHA256AndAES256CBC = 'PBESv2 using SHA256 PBKDF2 and AES256 CBC'