import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def _make_uint32(self, integer):
    """
        Create a 32 bit unsigned integer (The byte sequence of an integer).

        :param int integer: The integer value to convert
        :return: The byte sequence of an 32 bit integer
        """
    return struct.pack('!I', integer)