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
class SevenZipDecompressor:
    """Main decompressor object which is properly configured and bind to each 7zip folder.
    because 7zip folder can have a custom compression method"""

    def __init__(self, coders: List[Dict[str, Any]], packsize: int, unpacksizes: List[int], crc: Optional[int], password: Optional[str]=None, blocksize: Optional[int]=None) -> None:
        self.input_size = packsize
        self.unpacksizes = unpacksizes
        self.consumed: int = 0
        self.crc = crc
        self.digest: int = 0
        if blocksize:
            self.block_size: int = blocksize
        else:
            self.block_size = get_default_blocksize()
        if len(coders) > 4:
            raise UnsupportedCompressionMethodError(coders, 'Maximum cascade of filters is 4 but got {}.'.format(len(coders)))
        self.methods_map = [SupportedMethods.is_native_coder(coder) for coder in coders]
        if SupportedMethods.needs_password(coders) and password is None:
            raise PasswordRequired(coders, 'Password is required for extracting given archive.')
        if len(coders) >= 2:
            target_compressor = False
            has_bcj = False
            bcj_index = -1
            for i, coder in enumerate(coders):
                filter_id = SupportedMethods.get_filter_id(coder)
                if SupportedMethods.is_compressor_id(filter_id) and filter_id != FILTER_LZMA2:
                    target_compressor = True
                if filter_id in [FILTER_X86, FILTER_ARM, FILTER_ARMTHUMB, FILTER_POWERPC, FILTER_SPARC]:
                    has_bcj = True
                    bcj_index = i
                if target_compressor and has_bcj:
                    self.methods_map[bcj_index] = False
                    break
        self.chain = []
        self._unpacksizes = []
        self.input_size = self.input_size
        shift = 0
        prev = False
        for i, r in enumerate(self.methods_map):
            shift += 1 if r and prev else 0
            prev = r
            self._unpacksizes.append(unpacksizes[i - shift])
        self._unpacked = [0 for _ in range(len(self._unpacksizes))]
        self.consumed = 0
        self._unused = bytearray()
        self._buf = bytearray()
        self._pos = 0
        if all(self.methods_map):
            decompressor = self._get_lzma_decompressor(coders, unpacksizes[-1])
            self.chain.append(decompressor)
        elif not any(self.methods_map):
            for i in range(len(coders)):
                self.chain.append(self._get_alternative_decompressor(coders[i], unpacksizes[i], password))
        elif any(self.methods_map):
            for i in range(len(coders)):
                if not any(self.methods_map[:i]) and all(self.methods_map[i:]):
                    for j in range(i):
                        self.chain.append(self._get_alternative_decompressor(coders[j], unpacksizes[j], password))
                    self.chain.append(self._get_lzma_decompressor(coders[i:], unpacksizes[i]))
                    break
            else:
                for i in range(len(coders)):
                    if self.methods_map[i]:
                        self.chain.append(self._get_lzma_decompressor([coders[i]], unpacksizes[i]))
                    else:
                        self.chain.append(self._get_alternative_decompressor(coders[i], unpacksizes[i], password))
        else:
            raise UnsupportedCompressionMethodError(coders, 'Combination order of methods is not supported.')

    def _decompress(self, data, max_length: int):
        for i, decompressor in enumerate(self.chain):
            if self._unpacked[i] < self._unpacksizes[i]:
                data = decompressor.decompress(data, max_length)
                self._unpacked[i] += len(data)
            elif len(data) == 0:
                data = b''
            else:
                raise EOFError
        return data

    def _read_data(self, fp):
        rest_size = self.input_size - self.consumed
        unused_s = len(self._unused)
        read_size = min(rest_size - unused_s, self.block_size - unused_s)
        if read_size > 0:
            data = fp.read(read_size)
            self.consumed += len(data)
        else:
            data = b''
        return data

    def decompress(self, fp, max_length: int=-1) -> bytes:
        if max_length < 0:
            data = self._read_data(fp)
            res = self._buf[self._pos:] + self._decompress(self._unused + data, max_length)
            self._buf = bytearray()
            self._unused = bytearray()
            self._pos = 0
        else:
            current_buf_len = len(self._buf) - self._pos
            if current_buf_len >= max_length:
                res = self._buf[self._pos:self._pos + max_length]
                self._pos += max_length
            else:
                data = self._read_data(fp)
                if len(self._unused) > 0:
                    tmp = self._decompress(self._unused + data, max_length)
                    self._unused = bytearray()
                else:
                    tmp = self._decompress(data, max_length)
                if current_buf_len + len(tmp) <= max_length:
                    res = self._buf[self._pos:] + tmp
                    self._buf = bytearray()
                    self._pos = 0
                else:
                    res = self._buf[self._pos:] + tmp[:max_length - current_buf_len]
                    self._buf = bytearray(tmp[max_length - current_buf_len:])
                    self._pos = 0
        self.digest = calculate_crc32(res, self.digest)
        return res

    def check_crc(self):
        return self.crc == self.digest

    @property
    def unused_size(self):
        return len(self._unused)

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