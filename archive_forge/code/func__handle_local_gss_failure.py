import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _handle_local_gss_failure(self, e):
    self.transport.saved_exception = e
    self._log(DEBUG, 'GSSAPI failure: {}'.format(e))
    self._log(INFO, 'Authentication ({}) failed.'.format(self.auth_method))
    self.authenticated = False
    self.username = None
    if self.auth_event is not None:
        self.auth_event.set()
    return