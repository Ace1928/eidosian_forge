import logging
from .._compat import properties
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError, ExceptionRaisedContext
def _delete_password(self, target):
    try:
        win32cred.CredDelete(Type=win32cred.CRED_TYPE_GENERIC, TargetName=target)
    except pywintypes.error as e:
        if e.winerror == 1168 and e.funcname == 'CredDelete':
            return
        raise