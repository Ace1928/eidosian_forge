import bz2
import lzma
import struct
import sys
import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import bcj
import inflate64
import pyppmd
import pyzstd
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from py7zr.exceptions import PasswordRequired, UnsupportedCompressionMethodError
from py7zr.helpers import Buffer, calculate_crc32, calculate_key
from py7zr.properties import (
class BcjSparcDecoder(ISevenZipDecompressor):

    def __init__(self, size: int):
        self.decoder = bcj.SparcDecoder(size)

    def decompress(self, data: Union[bytes, bytearray, memoryview], max_length: int=-1) -> bytes:
        return self.decoder.decode(data)