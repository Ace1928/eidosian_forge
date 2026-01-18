import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexgss_error(self, m):
    """
        Parse the SSH2_MSG_KEXGSS_ERROR message (client mode).
        The server may send a GSS-API error message. if it does, we display
        the error by throwing an exception (client mode).

        :param `Message` m:  The content of the SSH2_MSG_KEXGSS_ERROR message
        :raise SSHException: Contains GSS-API major and minor status as well as
                             the error message and the language tag of the
                             message
        """
    maj_status = m.get_int()
    min_status = m.get_int()
    err_msg = m.get_string()
    m.get_string()
    raise SSHException('GSS-API Error:\nMajor Status: {}\nMinor Status: {}\nError Message: {}\n'.format(maj_status, min_status, err_msg))