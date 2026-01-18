import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher
import nacl.signing
from paramiko.message import Message
from paramiko.pkey import PKey, OPENSSH_AUTH_MAGIC, _unpad_openssh
from paramiko.util import b
from paramiko.ssh_exception import SSHException, PasswordRequiredException

    Representation of an `Ed25519 <https://ed25519.cr.yp.to/>`_ key.

    .. note::
        Ed25519 key support was added to OpenSSH in version 6.5.

    .. versionadded:: 2.2
    .. versionchanged:: 2.3
        Added a ``file_obj`` parameter to match other key classes.
    