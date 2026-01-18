import base64
import binascii
from abc import ABCMeta, abstractmethod
from typing import SupportsBytes, Type
class Encodable:

    def encode(self: SupportsBytes, encoder: Encoder=RawEncoder) -> bytes:
        return encoder.encode(bytes(self))