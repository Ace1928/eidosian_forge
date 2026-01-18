from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
class _OpensshWriter(object):
    """Writes SSH encoded values to a bytes-like buffer

    .. warning::
        This class is a private API and must not be exported outside of the openssh module_utils.
        It is not to be used to construct Openssh objects, but rather as a utility to assist
        in validating parsed material.
    """

    def __init__(self, buffer=None):
        if buffer is not None:
            if not isinstance(buffer, (bytes, bytearray)):
                raise TypeError('Buffer must be a bytes-like object not %s' % type(buffer))
        else:
            buffer = bytearray()
        self._buff = buffer

    def boolean(self, value):
        if not isinstance(value, bool):
            raise TypeError('Value must be of type bool not %s' % type(value))
        self._buff.extend(_BOOLEAN.pack(value))
        return self

    def uint32(self, value):
        if not isinstance(value, int):
            raise TypeError('Value must be of type int not %s' % type(value))
        if value < 0 or value > _UINT32_MAX:
            raise ValueError('Value must be a positive integer less than %s' % _UINT32_MAX)
        self._buff.extend(_UINT32.pack(value))
        return self

    def uint64(self, value):
        if not isinstance(value, (long, int)):
            raise TypeError('Value must be of type (long, int) not %s' % type(value))
        if value < 0 or value > _UINT64_MAX:
            raise ValueError('Value must be a positive integer less than %s' % _UINT64_MAX)
        self._buff.extend(_UINT64.pack(value))
        return self

    def string(self, value):
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError('Value must be bytes-like not %s' % type(value))
        self.uint32(len(value))
        self._buff.extend(value)
        return self

    def mpint(self, value):
        if not isinstance(value, (int, long)):
            raise TypeError('Value must be of type (long, int) not %s' % type(value))
        self.string(self._int_to_mpint(value))
        return self

    def name_list(self, value):
        if not isinstance(value, list):
            raise TypeError('Value must be a list of byte strings not %s' % type(value))
        try:
            self.string(','.join(value).encode('ASCII'))
        except UnicodeEncodeError as e:
            raise ValueError("Name-list's must consist of US-ASCII characters: %s" % e)
        return self

    def string_list(self, value):
        if not isinstance(value, list):
            raise TypeError('Value must be a list of byte string not %s' % type(value))
        writer = _OpensshWriter()
        for s in value:
            writer.string(s)
        self.string(writer.bytes())
        return self

    def option_list(self, value):
        if not isinstance(value, list) or (value and (not isinstance(value[0], tuple))):
            raise TypeError('Value must be a list of tuples')
        writer = _OpensshWriter()
        for name, data in value:
            writer.string(name)
            writer.string(_OpensshWriter().string(data).bytes() if data else bytes())
        self.string(writer.bytes())
        return self

    @staticmethod
    def _int_to_mpint(num):
        if PY3:
            byte_length = (num.bit_length() + 7) // 8
            try:
                result = num.to_bytes(byte_length, 'big', signed=True)
            except OverflowError:
                result = num.to_bytes(byte_length + 1, 'big', signed=True)
        else:
            result = bytes()
            if num == 0:
                result += b'\x00'
            elif num == -1:
                result += b'\xff'
            elif num > 0:
                while num >> 32:
                    result = _UINT32.pack(num & _UINT32_MAX) + result
                    num = num >> 32
                while num:
                    result = _UBYTE.pack(num & _UBYTE_MAX) + result
                    num = num >> 8
                if ord(result[0]) & 128:
                    result = b'\x00' + result
            else:
                while num >> 32 < -1:
                    result = _UINT32.pack(num & _UINT32_MAX) + result
                    num = num >> 32
                while num < -1:
                    result = _UBYTE.pack(num & _UBYTE_MAX) + result
                    num = num >> 8
                if not ord(result[0]) & 128:
                    result = b'\xff' + result
        return result

    def bytes(self):
        return bytes(self._buff)