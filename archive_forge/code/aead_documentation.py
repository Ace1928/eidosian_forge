from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag

    Checks whether the given cipher is supported through
    EVP_AEAD rather than the normal OpenSSL EVP_CIPHER API.
    