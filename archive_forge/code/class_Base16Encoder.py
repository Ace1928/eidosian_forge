import base64
import binascii
from abc import ABCMeta, abstractmethod
from typing import SupportsBytes, Type
class Base16Encoder(_Encoder):

    @staticmethod
    def encode(data: bytes) -> bytes:
        return base64.b16encode(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return base64.b16decode(data)