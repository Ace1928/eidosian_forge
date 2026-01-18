import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
class HexOutput(StaticClass):
    """
    Static functions for user output parsing.
    The counterparts for each method are in the L{HexInput} class.

    @type integer_size: int
    @cvar integer_size: Default size in characters of an outputted integer.
        This value is platform dependent.

    @type address_size: int
    @cvar address_size: Default Number of bits of the target architecture.
        This value is platform dependent.
    """
    integer_size = win32.SIZEOF(win32.DWORD) * 2 + 2
    address_size = win32.SIZEOF(win32.SIZE_T) * 2 + 2

    @classmethod
    def integer(cls, integer, bits=None):
        """
        @type  integer: int
        @param integer: Integer.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            integer_size = cls.integer_size
        else:
            integer_size = bits / 4 + 2
        if integer >= 0:
            return '0x%%.%dx' % (integer_size - 2) % integer
        return '-0x%%.%dx' % (integer_size - 2) % -integer

    @classmethod
    def address(cls, address, bits=None):
        """
        @type  address: int
        @param address: Memory address.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.address_size}

        @rtype:  str
        @return: Text output.
        """
        if bits is None:
            address_size = cls.address_size
            bits = win32.bits
        else:
            address_size = bits / 4 + 2
        if address < 0:
            address = 2 ** bits - 1 ^ ~address
        return '0x%%.%dx' % (address_size - 2) % address

    @staticmethod
    def hexadecimal(data):
        """
        Convert binary data to a string of hexadecimal numbers.

        @type  data: str
        @param data: Binary data.

        @rtype:  str
        @return: Hexadecimal representation.
        """
        return HexDump.hexadecimal(data, separator='')

    @classmethod
    def integer_list_file(cls, filename, values, bits=None):
        """
        Write a list of integers to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.integer_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of integers to write to the file.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}
        """
        fd = open(filename, 'w')
        for integer in values:
            (print >> fd, cls.integer(integer, bits))
        fd.close()

    @classmethod
    def string_list_file(cls, filename, values):
        """
        Write a list of strings to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.string_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of strings to write to the file.
        """
        fd = open(filename, 'w')
        for string in values:
            (print >> fd, string)
        fd.close()

    @classmethod
    def mixed_list_file(cls, filename, values, bits):
        """
        Write a list of mixed values to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.mixed_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of mixed values to write to the file.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}
        """
        fd = open(filename, 'w')
        for original in values:
            try:
                parsed = cls.integer(original, bits)
            except TypeError:
                parsed = repr(original)
            (print >> fd, parsed)
        fd.close()