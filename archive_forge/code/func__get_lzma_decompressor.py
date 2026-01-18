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
def _get_lzma_decompressor(self, coders: List[Dict[str, Any]], unpacksize: int):
    filters: List[Dict[str, Any]] = []
    lzma1 = False
    for coder in coders:
        if coder['numinstreams'] != 1 or coder['numoutstreams'] != 1:
            raise UnsupportedCompressionMethodError(coders, 'Only a simple compression method is currently supported.')
        if not SupportedMethods.is_native_coder(coder):
            raise UnsupportedCompressionMethodError(coders, 'Non python native method is requested.')
        properties = coder.get('properties', None)
        filter_id = SupportedMethods.get_filter_id(coder)
        if filter_id == FILTER_LZMA:
            lzma1 = True
        if properties is not None:
            filters[:0] = [lzma._decode_filter_properties(filter_id, properties)]
        else:
            filters[:0] = [{'id': filter_id}]
    if lzma1:
        return LZMA1Decompressor(filters, unpacksize)
    else:
        return lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)