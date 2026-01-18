from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class HashAlgorithm(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        A string naming this algorithm (e.g. "sha256", "md5").
        """

    @property
    @abc.abstractmethod
    def digest_size(self) -> int:
        """
        The size of the resulting digest in bytes.
        """

    @property
    @abc.abstractmethod
    def block_size(self) -> typing.Optional[int]:
        """
        The internal block size of the hash function, or None if the hash
        function does not use blocks internally (e.g. SHA3).
        """