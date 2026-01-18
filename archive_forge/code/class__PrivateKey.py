import os.path
import secrets
import ssl
import tempfile
import typing as t
class _PrivateKey(t.Protocol):

    def private_bytes(self, encoding: 'serialization.Encoding', format: 'serialization.PrivateFormat', encryption_algorithm: 'serialization.KeySerializationEncryption') -> bytes:
        ...