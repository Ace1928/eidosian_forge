import os
from ctypes import *
def _logoff(session):
    rc = MAPILogoff(session, 0, 0, 0)
    if rc != SUCCESS_SUCCESS:
        raise MAPIError(rc)