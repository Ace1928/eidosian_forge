import base64
import binascii
from abc import ABCMeta, abstractmethod
from typing import SupportsBytes, Type
class RawEncoder(_Encoder):

    @staticmethod
    def encode(data: bytes) -> bytes:
        return data

    @staticmethod
    def decode(data: bytes) -> bytes:
        return data