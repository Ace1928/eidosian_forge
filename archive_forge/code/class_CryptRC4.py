import secrets
from Crypto import __version__
from Crypto.Cipher import AES, ARC4
from Crypto.Util.Padding import pad
from pypdf._crypt_providers._base import CryptBase
class CryptRC4(CryptBase):

    def __init__(self, key: bytes) -> None:
        self.key = key

    def encrypt(self, data: bytes) -> bytes:
        return ARC4.ARC4Cipher(self.key).encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return ARC4.ARC4Cipher(self.key).decrypt(data)