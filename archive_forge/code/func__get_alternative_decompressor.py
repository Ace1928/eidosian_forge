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
def _get_alternative_decompressor(self, coder: Dict[str, Any], unpacksize=None, password=None) -> Union[bz2.BZ2Decompressor, lzma.LZMADecompressor, ISevenZipDecompressor]:
    filter_id = SupportedMethods.get_filter_id(coder)
    if filter_id in [FILTER_X86, FILTER_ARM, FILTER_ARMTHUMB, FILTER_POWERPC, FILTER_SPARC]:
        return algorithm_class_map[filter_id][1](size=unpacksize)
    if SupportedMethods.is_native_coder(coder):
        raise UnsupportedCompressionMethodError(coder, 'Unknown method code:{}'.format(coder['method']))
    if filter_id not in algorithm_class_map:
        raise UnsupportedCompressionMethodError(coder, 'Unknown method filter_id:{}'.format(filter_id))
    if algorithm_class_map[filter_id][1] is None:
        raise UnsupportedCompressionMethodError(coder, 'Decompression is not supported by {}.'.format(SupportedMethods.get_method_name_id(filter_id)))
    if SupportedMethods.is_crypto_id(filter_id):
        return algorithm_class_map[filter_id][1](coder['properties'], password, self.block_size)
    elif SupportedMethods.need_property(filter_id):
        return algorithm_class_map[filter_id][1](coder['properties'], self.block_size)
    else:
        return algorithm_class_map[filter_id][1]()