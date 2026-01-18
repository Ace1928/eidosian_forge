import os
import socket
import struct
import sys
import threading
import time
import tempfile
import stat
from logging import DEBUG
from select import select
from paramiko.common import io_sleep, byte_chr
from paramiko.ssh_exception import SSHException, AuthenticationException
from paramiko.message import Message
from paramiko.pkey import PKey, UnknownKeyType
from paramiko.util import asbytes, get_logger
class AgentKey(PKey):
    """
    Private key held in a local SSH agent.  This type of key can be used for
    authenticating to a remote server (signing).  Most other key operations
    work as expected.

    .. versionchanged:: 3.2
        Added the ``comment`` kwarg and attribute.

    .. versionchanged:: 3.2
        Added the ``.inner_key`` attribute holding a reference to the 'real'
        key instance this key is a proxy for, if one was obtainable, else None.
    """

    def __init__(self, agent, blob, comment=''):
        self.agent = agent
        self.blob = blob
        self.comment = comment
        msg = Message(blob)
        self.name = msg.get_text()
        self._logger = get_logger(__file__)
        self.inner_key = None
        try:
            self.inner_key = PKey.from_type_string(key_type=self.name, key_bytes=blob)
        except UnknownKeyType:
            err = 'Unable to derive inner_key for agent key of type {!r}'
            self.log(DEBUG, err.format(self.name))

    def log(self, *args, **kwargs):
        return self._logger.log(*args, **kwargs)

    def asbytes(self):
        return self.inner_key.asbytes() if self.inner_key else self.blob

    def get_name(self):
        return self.name

    def get_bits(self):
        if self.inner_key is not None:
            return self.inner_key.get_bits()
        return super().get_bits()

    def __getattr__(self, name):
        """
        Proxy any un-implemented methods/properties to the inner_key.
        """
        if self.inner_key is None:
            raise AttributeError(name)
        return getattr(self.inner_key, name)

    @property
    def _fields(self):
        fallback = [self.get_name(), self.blob]
        return self.inner_key._fields if self.inner_key else fallback

    def sign_ssh_data(self, data, algorithm=None):
        msg = Message()
        msg.add_byte(cSSH2_AGENTC_SIGN_REQUEST)
        msg.add_string(self.asbytes())
        msg.add_string(data)
        msg.add_int(ALGORITHM_FLAG_MAP.get(algorithm, 0))
        ptype, result = self.agent._send_message(msg)
        if ptype != SSH2_AGENT_SIGN_RESPONSE:
            raise SSHException('key cannot be used for signing')
        return result.get_binary()