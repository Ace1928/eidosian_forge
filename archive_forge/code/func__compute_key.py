import os
import socket
import sys
import threading
import time
import weakref
from hashlib import md5, sha1, sha256, sha512
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes
import paramiko
from paramiko import util
from paramiko.auth_handler import AuthHandler, AuthOnlyHandler
from paramiko.ssh_gss import GSSAuth
from paramiko.channel import Channel
from paramiko.common import (
from paramiko.compress import ZlibCompressor, ZlibDecompressor
from paramiko.dsskey import DSSKey
from paramiko.ed25519key import Ed25519Key
from paramiko.kex_curve25519 import KexCurve25519
from paramiko.kex_gex import KexGex, KexGexSHA256
from paramiko.kex_group1 import KexGroup1
from paramiko.kex_group14 import KexGroup14, KexGroup14SHA256
from paramiko.kex_group16 import KexGroup16SHA512
from paramiko.kex_ecdh_nist import KexNistp256, KexNistp384, KexNistp521
from paramiko.kex_gss import KexGSSGex, KexGSSGroup1, KexGSSGroup14
from paramiko.message import Message
from paramiko.packet import Packetizer, NeedRekeyException
from paramiko.primes import ModulusPack
from paramiko.rsakey import RSAKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.server import ServerInterface
from paramiko.sftp_client import SFTPClient
from paramiko.ssh_exception import (
from paramiko.util import (
import atexit
def _compute_key(self, id, nbytes):
    """id is 'A' - 'F' for the various keys used by ssh"""
    m = Message()
    m.add_mpint(self.K)
    m.add_bytes(self.H)
    m.add_byte(b(id))
    m.add_bytes(self.session_id)
    hash_algo = getattr(self.kex_engine, 'hash_algo', None)
    hash_select_msg = 'kex engine {} specified hash_algo {!r}'.format(self.kex_engine.__class__.__name__, hash_algo)
    if hash_algo is None:
        hash_algo = sha1
        hash_select_msg += ', falling back to sha1'
    if not hasattr(self, '_logged_hash_selection'):
        self._log(DEBUG, hash_select_msg)
        setattr(self, '_logged_hash_selection', True)
    out = sofar = hash_algo(m.asbytes()).digest()
    while len(out) < nbytes:
        m = Message()
        m.add_mpint(self.K)
        m.add_bytes(self.H)
        m.add_bytes(sofar)
        digest = hash_algo(m.asbytes()).digest()
        out += digest
        sofar += digest
    return out[:nbytes]