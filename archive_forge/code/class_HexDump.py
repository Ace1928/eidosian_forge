import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
class HexDump(StaticClass):
    """
    Static functions for hexadecimal dumps.

    @type integer_size: int
    @cvar integer_size: Size in characters of an outputted integer.
        This value is platform dependent.

    @type address_size: int
    @cvar address_size: Size in characters of an outputted address.
        This value is platform dependent.
    """
    integer_size = win32.SIZEOF(win32.DWORD) * 2
    address_size = win32.SIZEOF(win32.SIZE_T) * 2

    @classmethod
    def integer(cls, integer, bits=None):
        """
        @type  integer: int
        @param integer: Integer.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.integer_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            integer_size = cls.integer_size
        else:
            integer_size = bits / 4
        return '%%.%dX' % integer_size % integer

    @classmethod
    def address(cls, address, bits=None):
        """
        @type  address: int
        @param address: Memory address.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            address_size = cls.address_size
            bits = win32.bits
        else:
            address_size = bits / 4
        if address < 0:
            address = 2 ** bits - 1 ^ ~address
        return '%%.%dX' % address_size % address

    @staticmethod
    def printable(data):
        """
        Replace unprintable characters with dots.

        @type  data: str
        @param data: Binary data.

        @rtype:  str
        @return: Printable text.
        """
        result = ''
        for c in data:
            if 32 < ord(c) < 128:
                result += c
            else:
                result += '.'
        return result

    @staticmethod
    def hexadecimal(data, separator=''):
        """
        Convert binary data to a string of hexadecimal numbers.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        return separator.join(['%.2x' % ord(c) for c in data])

    @staticmethod
    def hexa_word(data, separator=' '):
        """
        Convert binary data to a string of hexadecimal WORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each WORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        if len(data) & 1 != 0:
            data += '\x00'
        return separator.join(['%.4x' % struct.unpack('<H', data[i:i + 2])[0] for i in compat.xrange(0, len(data), 2)])

    @staticmethod
    def hexa_dword(data, separator=' '):
        """
        Convert binary data to a string of hexadecimal DWORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each DWORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        if len(data) & 3 != 0:
            data += '\x00' * (4 - (len(data) & 3))
        return separator.join(['%.8x' % struct.unpack('<L', data[i:i + 4])[0] for i in compat.xrange(0, len(data), 4)])

    @staticmethod
    def hexa_qword(data, separator=' '):
        """
        Convert binary data to a string of hexadecimal QWORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each QWORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        if len(data) & 7 != 0:
            data += '\x00' * (8 - (len(data) & 7))
        return separator.join(['%.16x' % struct.unpack('<Q', data[i:i + 8])[0] for i in compat.xrange(0, len(data), 8)])

    @classmethod
    def hexline(cls, data, separator=' ', width=None):
        """
        Dump a line of hexadecimal numbers from binary data.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.
            This value is also used for padding.

        @rtype:  str
        @return: Multiline output text.
        """
        if width is None:
            fmt = '%s  %s'
        else:
            fmt = '%%-%ds  %%-%ds' % ((len(separator) + 2) * width - 1, width)
        return fmt % (cls.hexadecimal(data, separator), cls.printable(data))

    @classmethod
    def hexblock(cls, data, address=None, bits=None, separator=' ', width=8):
        """
        Dump a block of hexadecimal numbers from binary data.
        Also show a printable text version of the data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexline, data, address, bits, width, cb_kwargs={'width': width, 'separator': separator})

    @classmethod
    def hexblock_cb(cls, callback, data, address=None, bits=None, width=16, cb_args=(), cb_kwargs={}):
        """
        Dump a block of binary data using a callback function to convert each
        line of text.

        @type  callback: function
        @param callback: Callback function to convert each line of data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address:
            (Optional) Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  cb_args: str
        @param cb_args:
            (Optional) Arguments to pass to the callback function.

        @type  cb_kwargs: str
        @param cb_kwargs:
            (Optional) Keyword arguments to pass to the callback function.

        @type  width: int
        @param width:
            (Optional) Maximum number of bytes to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        result = ''
        if address is None:
            for i in compat.xrange(0, len(data), width):
                result = '%s%s\n' % (result, callback(data[i:i + width], *cb_args, **cb_kwargs))
        else:
            for i in compat.xrange(0, len(data), width):
                result = '%s%s: %s\n' % (result, cls.address(address, bits), callback(data[i:i + width], *cb_args, **cb_kwargs))
                address += width
        return result

    @classmethod
    def hexblock_byte(cls, data, address=None, bits=None, separator=' ', width=16):
        """
        Dump a block of hexadecimal BYTEs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each BYTE.

        @type  width: int
        @param width:
            (Optional) Maximum number of BYTEs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexadecimal, data, address, bits, width, cb_kwargs={'separator': separator})

    @classmethod
    def hexblock_word(cls, data, address=None, bits=None, separator=' ', width=8):
        """
        Dump a block of hexadecimal WORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each WORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of WORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexa_word, data, address, bits, width * 2, cb_kwargs={'separator': separator})

    @classmethod
    def hexblock_dword(cls, data, address=None, bits=None, separator=' ', width=4):
        """
        Dump a block of hexadecimal DWORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each DWORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of DWORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexa_dword, data, address, bits, width * 4, cb_kwargs={'separator': separator})

    @classmethod
    def hexblock_qword(cls, data, address=None, bits=None, separator=' ', width=2):
        """
        Dump a block of hexadecimal QWORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each QWORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of QWORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
        return cls.hexblock_cb(cls.hexa_qword, data, address, bits, width * 8, cb_kwargs={'separator': separator})