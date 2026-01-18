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
def _parse_service_request(self, m):
    self._restore_delegate_auth_handler()
    return self._delegate._parse_service_request(m)