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
@property
def _client_handler_table(self):
    my_table = super()._client_handler_table.copy()
    del my_table[MSG_SERVICE_ACCEPT]
    return my_table