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
def _key_from_filepath(self, filename, klass, password):
    """
        Attempt to derive a `.PKey` from given string path ``filename``:

        - If ``filename`` appears to be a cert, the matching private key is
          loaded.
        - Otherwise, the filename is assumed to be a private key, and the
          matching public cert will be loaded if it exists.
        """
    cert_suffix = '-cert.pub'
    if filename.endswith(cert_suffix):
        key_path = filename[:-len(cert_suffix)]
        cert_path = filename
    else:
        key_path = filename
        cert_path = filename + cert_suffix
    key = klass.from_private_key_file(key_path, password)
    msg = 'Trying discovered key {} in {}'.format(hexlify(key.get_fingerprint()), key_path)
    self._log(DEBUG, msg)
    if os.path.isfile(cert_path):
        key.load_certificate(cert_path)
        self._log(DEBUG, 'Adding public certificate {}'.format(cert_path))
    return key