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
def _send_kex_init(self):
    """
        announce to the other side that we'd like to negotiate keys, and what
        kind of key negotiation we support.
        """
    self.clear_to_send_lock.acquire()
    try:
        self.clear_to_send.clear()
    finally:
        self.clear_to_send_lock.release()
    self.gss_kex_used = False
    self.in_kex = True
    kex_algos = list(self.preferred_kex)
    if self.server_mode:
        mp_required_prefix = 'diffie-hellman-group-exchange-sha'
        kex_mp = [k for k in kex_algos if k.startswith(mp_required_prefix)]
        if self._modulus_pack is None and len(kex_mp) > 0:
            pkex = [k for k in self.get_security_options().kex if not k.startswith(mp_required_prefix)]
            self.get_security_options().kex = pkex
        available_server_keys = list(filter(list(self.server_key_dict.keys()).__contains__, self.preferred_keys))
    else:
        available_server_keys = self.preferred_keys
        kex_algos.append('ext-info-c')
    if self.advertise_strict_kex:
        which = 's' if self.server_mode else 'c'
        kex_algos.append(f'kex-strict-{which}-v00@openssh.com')
    m = Message()
    m.add_byte(cMSG_KEXINIT)
    m.add_bytes(os.urandom(16))
    m.add_list(kex_algos)
    m.add_list(available_server_keys)
    m.add_list(self.preferred_ciphers)
    m.add_list(self.preferred_ciphers)
    m.add_list(self.preferred_macs)
    m.add_list(self.preferred_macs)
    m.add_list(self.preferred_compression)
    m.add_list(self.preferred_compression)
    m.add_string(bytes())
    m.add_string(bytes())
    m.add_boolean(False)
    m.add_int(0)
    self.local_kex_init = self._latest_kex_init = m.asbytes()
    self._send_message(m)