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
def _activate_inbound(self):
    """switch on newly negotiated encryption parameters for
        inbound traffic"""
    block_size = self._cipher_info[self.remote_cipher]['block-size']
    if self.server_mode:
        IV_in = self._compute_key('A', block_size)
        key_in = self._compute_key('C', self._cipher_info[self.remote_cipher]['key-size'])
    else:
        IV_in = self._compute_key('B', block_size)
        key_in = self._compute_key('D', self._cipher_info[self.remote_cipher]['key-size'])
    engine = self._get_cipher(self.remote_cipher, key_in, IV_in, self._DECRYPT)
    etm = 'etm@openssh.com' in self.remote_mac
    mac_size = self._mac_info[self.remote_mac]['size']
    mac_engine = self._mac_info[self.remote_mac]['class']
    if self.server_mode:
        mac_key = self._compute_key('E', mac_engine().digest_size)
    else:
        mac_key = self._compute_key('F', mac_engine().digest_size)
    self.packetizer.set_inbound_cipher(engine, block_size, mac_engine, mac_size, mac_key, etm=etm)
    compress_in = self._compression_info[self.remote_compression][1]
    if compress_in is not None and (self.remote_compression != 'zlib@openssh.com' or self.authenticated):
        self._log(DEBUG, 'Switching on inbound compression ...')
        self.packetizer.set_inbound_compressor(compress_in())
    if self.agreed_on_strict_kex:
        self._log(DEBUG, 'Resetting inbound seqno after NEWKEYS due to strict mode')
        self.packetizer.reset_seqno_in()