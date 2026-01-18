import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def _ssh_build_mic(self, session_id, username, service, auth_method):
    """
        Create the SSH2 MIC filed for gssapi-with-mic.

        :param str session_id: The SSH session ID
        :param str username: The name of the user who attempts to login
        :param str service: The requested SSH service
        :param str auth_method: The requested SSH authentication mechanism
        :return: The MIC as defined in RFC 4462. The contents of the
                 MIC field are:
                 string    session_identifier,
                 byte      SSH_MSG_USERAUTH_REQUEST,
                 string    user-name,
                 string    service (ssh-connection),
                 string    authentication-method
                           (gssapi-with-mic or gssapi-keyex)
        """
    mic = self._make_uint32(len(session_id))
    mic += session_id
    mic += struct.pack('B', MSG_USERAUTH_REQUEST)
    mic += self._make_uint32(len(username))
    mic += username.encode()
    mic += self._make_uint32(len(service))
    mic += service.encode()
    mic += self._make_uint32(len(auth_method))
    mic += auth_method.encode()
    return mic