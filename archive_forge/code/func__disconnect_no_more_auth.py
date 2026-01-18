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
def _disconnect_no_more_auth(self):
    m = Message()
    m.add_byte(cMSG_DISCONNECT)
    m.add_int(DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE)
    m.add_string('No more auth methods available')
    m.add_string('en')
    self.transport._send_message(m)
    self.transport.close()