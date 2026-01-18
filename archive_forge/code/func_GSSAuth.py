import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def GSSAuth(auth_method, gss_deleg_creds=True):
    """
    Provide SSH2 GSS-API / SSPI authentication.

    :param str auth_method: The name of the SSH authentication mechanism
                            (gssapi-with-mic or gss-keyex)
    :param bool gss_deleg_creds: Delegate client credentials or not.
                                 We delegate credentials by default.
    :return: Either an `._SSH_GSSAPI_OLD` or `._SSH_GSSAPI_NEW` (Unix)
             object or an `_SSH_SSPI` (Windows) object
    :rtype: object

    :raises: ``ImportError`` -- If no GSS-API / SSPI module could be imported.

    :see: `RFC 4462 <http://www.ietf.org/rfc/rfc4462.txt>`_
    :note: Check for the available API and return either an `._SSH_GSSAPI_OLD`
           (MIT GSSAPI using python-gssapi package) object, an
           `._SSH_GSSAPI_NEW` (MIT GSSAPI using gssapi package) object
           or an `._SSH_SSPI` (MS SSPI) object.
           If there is no supported API available,
           ``None`` will be returned.
    """
    if _API == 'MIT':
        return _SSH_GSSAPI_OLD(auth_method, gss_deleg_creds)
    elif _API == 'PYTHON-GSSAPI-NEW':
        return _SSH_GSSAPI_NEW(auth_method, gss_deleg_creds)
    elif _API == 'SSPI' and os.name == 'nt':
        return _SSH_SSPI(auth_method, gss_deleg_creds)
    else:
        raise ImportError('Unable to import a GSS-API / SSPI module!')