from __future__ import annotations
import abc
import typing
class BlockCipherAlgorithm(CipherAlgorithm):
    key: bytes

    @property
    @abc.abstractmethod
    def block_size(self) -> int:
        """
        The size of a block as an integer in bits (e.g. 64, 128).
        """