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
def _interactive_query(self, q):
    m = Message()
    m.add_byte(cMSG_USERAUTH_INFO_REQUEST)
    m.add_string(q.name)
    m.add_string(q.instructions)
    m.add_string(bytes())
    m.add_int(len(q.prompts))
    for p in q.prompts:
        m.add_string(p[0])
        m.add_boolean(p[1])
    self.transport._send_message(m)