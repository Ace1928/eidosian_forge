import base64
import binascii
from abc import ABCMeta, abstractmethod
from typing import SupportsBytes, Type
class HexEncoder(_Encoder):

    @staticmethod
    def encode(data: bytes) -> bytes:
        return binascii.hexlify(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return binascii.unhexlify(data)