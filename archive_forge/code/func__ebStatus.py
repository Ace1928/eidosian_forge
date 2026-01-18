import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def _ebStatus(self, reason, requestId, msg=b'request failed'):
    code = FX_FAILURE
    message = msg
    if isinstance(reason.value, (IOError, OSError)):
        if reason.value.errno == errno.ENOENT:
            code = FX_NO_SUCH_FILE
            message = networkString(reason.value.strerror)
        elif reason.value.errno == errno.EACCES:
            code = FX_PERMISSION_DENIED
            message = networkString(reason.value.strerror)
        elif reason.value.errno == errno.EEXIST:
            code = FX_FILE_ALREADY_EXISTS
        else:
            self._log.failure('Request {requestId} failed: {message}', failure=reason, requestId=requestId, message=message)
    elif isinstance(reason.value, EOFError):
        code = FX_EOF
        if reason.value.args:
            message = networkString(reason.value.args[0])
    elif isinstance(reason.value, NotImplementedError):
        code = FX_OP_UNSUPPORTED
        if reason.value.args:
            message = networkString(reason.value.args[0])
    elif isinstance(reason.value, SFTPError):
        code = reason.value.code
        message = networkString(reason.value.message)
    else:
        self._log.failure('Request {requestId} failed with unknown error: {message}', failure=reason, requestId=requestId, message=message)
    self._sendStatus(requestId, code, message)