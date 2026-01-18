import abc
import os
class MemoryKeyStore(RootKeyStore):
    """ MemoryKeyStore returns an implementation of
    Store that generates a single key and always
    returns that from root_key. The same id ("0") is always
    used.
    """

    def __init__(self, key=None):
        """ If the key is not specified a random key will be generated.
        @param key: bytes
        """
        if key is None:
            key = os.urandom(24)
        self._key = key

    def get(self, id):
        if id != b'0':
            return None
        return self._key

    def root_key(self):
        return (self._key, b'0')