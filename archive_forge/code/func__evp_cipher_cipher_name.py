from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_cipher_name(cipher: _AEADTypes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESCCM, AESGCM, AESOCB3, AESSIV, ChaCha20Poly1305
    if isinstance(cipher, ChaCha20Poly1305):
        return b'chacha20-poly1305'
    elif isinstance(cipher, AESCCM):
        return f'aes-{len(cipher._key) * 8}-ccm'.encode('ascii')
    elif isinstance(cipher, AESOCB3):
        return f'aes-{len(cipher._key) * 8}-ocb'.encode('ascii')
    elif isinstance(cipher, AESSIV):
        return f'aes-{len(cipher._key) * 8 // 2}-siv'.encode('ascii')
    else:
        assert isinstance(cipher, AESGCM)
        return f'aes-{len(cipher._key) * 8}-gcm'.encode('ascii')