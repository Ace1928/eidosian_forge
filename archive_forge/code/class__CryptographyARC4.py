import struct
class _CryptographyARC4(object):

    def __init__(self, key):
        algo = algorithms.ARC4(key)
        cipher = Cipher(algo, mode=None, backend=default_backend())
        self._encryptor = cipher.encryptor()

    def update(self, value):
        return self._encryptor.update(value)