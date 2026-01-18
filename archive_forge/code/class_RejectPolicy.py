from binascii import hexlify
import getpass
import inspect
import os
import socket
import warnings
from errno import ECONNREFUSED, EHOSTUNREACH
from paramiko.agent import Agent
from paramiko.common import DEBUG
from paramiko.config import SSH_PORT
from paramiko.dsskey import DSSKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.ed25519key import Ed25519Key
from paramiko.hostkeys import HostKeys
from paramiko.rsakey import RSAKey
from paramiko.ssh_exception import (
from paramiko.transport import Transport
from paramiko.util import ClosingContextManager
class RejectPolicy(MissingHostKeyPolicy):
    """
    Policy for automatically rejecting the unknown hostname & key.  This is
    used by `.SSHClient`.
    """

    def missing_host_key(self, client, hostname, key):
        client._log(DEBUG, 'Rejecting {} host key for {}: {}'.format(key.get_name(), hostname, hexlify(key.get_fingerprint())))
        raise SSHException('Server {!r} not found in known_hosts'.format(hostname))