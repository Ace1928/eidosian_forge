import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
class HexInput(StaticClass):
    """
    Static functions for user input parsing.
    The counterparts for each method are in the L{HexOutput} class.
    """

    @staticmethod
    def integer(token):
        """
        Convert numeric strings into integers.

        @type  token: str
        @param token: String to parse.

        @rtype:  int
        @return: Parsed integer value.
        """
        token = token.strip()
        neg = False
        if token.startswith(compat.b('-')):
            token = token[1:]
            neg = True
        if token.startswith(compat.b('0x')):
            result = int(token, 16)
        elif token.startswith(compat.b('0b')):
            result = int(token[2:], 2)
        elif token.startswith(compat.b('0o')):
            result = int(token, 8)
        else:
            try:
                result = int(token)
            except ValueError:
                result = int(token, 16)
        if neg:
            result = -result
        return result

    @staticmethod
    def address(token):
        """
        Convert numeric strings into memory addresses.

        @type  token: str
        @param token: String to parse.

        @rtype:  int
        @return: Parsed integer value.
        """
        return int(token, 16)

    @staticmethod
    def hexadecimal(token):
        """
        Convert a strip of hexadecimal numbers into binary data.

        @type  token: str
        @param token: String to parse.

        @rtype:  str
        @return: Parsed string value.
        """
        token = ''.join([c for c in token if c.isalnum()])
        if len(token) % 2 != 0:
            raise ValueError('Missing characters in hex data')
        data = ''
        for i in compat.xrange(0, len(token), 2):
            x = token[i:i + 2]
            d = int(x, 16)
            s = struct.pack('<B', d)
            data += s
        return data

    @staticmethod
    def pattern(token):
        """
        Convert an hexadecimal search pattern into a POSIX regular expression.

        For example, the following pattern::

            "B8 0? ?0 ?? ??"

        Would match the following data::

            "B8 0D F0 AD BA"    # mov eax, 0xBAADF00D

        @type  token: str
        @param token: String to parse.

        @rtype:  str
        @return: Parsed string value.
        """
        token = ''.join([c for c in token if c == '?' or c.isalnum()])
        if len(token) % 2 != 0:
            raise ValueError('Missing characters in hex data')
        regexp = ''
        for i in compat.xrange(0, len(token), 2):
            x = token[i:i + 2]
            if x == '??':
                regexp += '.'
            elif x[0] == '?':
                f = '\\x%%.1x%s' % x[1]
                x = ''.join([f % c for c in compat.xrange(0, 16)])
                regexp = '%s[%s]' % (regexp, x)
            elif x[1] == '?':
                f = '\\x%s%%.1x' % x[0]
                x = ''.join([f % c for c in compat.xrange(0, 16)])
                regexp = '%s[%s]' % (regexp, x)
            else:
                regexp = '%s\\x%s' % (regexp, x)
        return regexp

    @staticmethod
    def is_pattern(token):
        """
        Determine if the given argument is a valid hexadecimal pattern to be
        used with L{pattern}.

        @type  token: str
        @param token: String to parse.

        @rtype:  bool
        @return:
            C{True} if it's a valid hexadecimal pattern, C{False} otherwise.
        """
        return re.match('^(?:[\\?A-Fa-f0-9][\\?A-Fa-f0-9]\\s*)+$', token)

    @classmethod
    def integer_list_file(cls, filename):
        """
        Read a list of integers from a file.

        The file format is:

         - # anywhere in the line begins a comment
         - leading and trailing spaces are ignored
         - empty lines are ignored
         - integers can be specified as:
            - decimal numbers ("100" is 100)
            - hexadecimal numbers ("0x100" is 256)
            - binary numbers ("0b100" is 4)
            - octal numbers ("0100" is 64)

        @type  filename: str
        @param filename: Name of the file to read.

        @rtype:  list( int )
        @return: List of integers read from the file.
        """
        count = 0
        result = list()
        fd = open(filename, 'r')
        for line in fd:
            count = count + 1
            if '#' in line:
                line = line[:line.find('#')]
            line = line.strip()
            if line:
                try:
                    value = cls.integer(line)
                except ValueError:
                    e = sys.exc_info()[1]
                    msg = 'Error in line %d of %s: %s'
                    msg = msg % (count, filename, str(e))
                    raise ValueError(msg)
                result.append(value)
        return result

    @classmethod
    def string_list_file(cls, filename):
        """
        Read a list of string values from a file.

        The file format is:

         - # anywhere in the line begins a comment
         - leading and trailing spaces are ignored
         - empty lines are ignored
         - strings cannot span over a single line

        @type  filename: str
        @param filename: Name of the file to read.

        @rtype:  list
        @return: List of integers and strings read from the file.
        """
        count = 0
        result = list()
        fd = open(filename, 'r')
        for line in fd:
            count = count + 1
            if '#' in line:
                line = line[:line.find('#')]
            line = line.strip()
            if line:
                result.append(line)
        return result

    @classmethod
    def mixed_list_file(cls, filename):
        """
        Read a list of mixed values from a file.

        The file format is:

         - # anywhere in the line begins a comment
         - leading and trailing spaces are ignored
         - empty lines are ignored
         - strings cannot span over a single line
         - integers can be specified as:
            - decimal numbers ("100" is 100)
            - hexadecimal numbers ("0x100" is 256)
            - binary numbers ("0b100" is 4)
            - octal numbers ("0100" is 64)

        @type  filename: str
        @param filename: Name of the file to read.

        @rtype:  list
        @return: List of integers and strings read from the file.
        """
        count = 0
        result = list()
        fd = open(filename, 'r')
        for line in fd:
            count = count + 1
            if '#' in line:
                line = line[:line.find('#')]
            line = line.strip()
            if line:
                try:
                    value = cls.integer(line)
                except ValueError:
                    value = line
                result.append(value)
        return result