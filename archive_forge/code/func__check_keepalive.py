import errno
import os
import socket
import struct
import threading
import time
from hmac import HMAC
from paramiko import util
from paramiko.common import (
from paramiko.util import u
from paramiko.ssh_exception import SSHException, ProxyCommandFailure
from paramiko.message import Message
def _check_keepalive(self):
    if not self.__keepalive_interval or not self.__block_engine_out or self.__need_rekey:
        return
    now = time.time()
    if now > self.__keepalive_last + self.__keepalive_interval:
        self.__keepalive_callback()
        self.__keepalive_last = now