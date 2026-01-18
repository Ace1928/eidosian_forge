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
def _choose_fallback_pubkey_algorithm(self, key_type, my_algos):
    msg = 'Server did not send a server-sig-algs list; defaulting to something in our preferred algorithms list'
    self._log(DEBUG, msg)
    noncert_key_type = key_type.replace('-cert-v01@openssh.com', '')
    if key_type in my_algos or noncert_key_type in my_algos:
        actual = key_type if key_type in my_algos else noncert_key_type
        msg = f'Current key type, {actual!r}, is in our preferred list; using that'
        algo = actual
    else:
        algo = my_algos[0]
        msg = f'{key_type!r} not in our list - trying first list item instead, {algo!r}'
    self._log(DEBUG, msg)
    return algo