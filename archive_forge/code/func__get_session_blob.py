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
def _get_session_blob(self, key, service, username, algorithm):
    m = Message()
    m.add_string(self.transport.session_id)
    m.add_byte(cMSG_USERAUTH_REQUEST)
    m.add_string(username)
    m.add_string(service)
    m.add_string('publickey')
    m.add_boolean(True)
    _, bits = self._get_key_type_and_bits(key)
    m.add_string(algorithm)
    m.add_string(bits)
    return m.asbytes()