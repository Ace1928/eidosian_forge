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
def _parse_kex_init(self, m):
    parsed = self._really_parse_kex_init(m)
    kex_algo_list = parsed['kex_algo_list']
    server_key_algo_list = parsed['server_key_algo_list']
    client_encrypt_algo_list = parsed['client_encrypt_algo_list']
    server_encrypt_algo_list = parsed['server_encrypt_algo_list']
    client_mac_algo_list = parsed['client_mac_algo_list']
    server_mac_algo_list = parsed['server_mac_algo_list']
    client_compress_algo_list = parsed['client_compress_algo_list']
    server_compress_algo_list = parsed['server_compress_algo_list']
    client_lang_list = parsed['client_lang_list']
    server_lang_list = parsed['server_lang_list']
    kex_follows = parsed['kex_follows']
    self._log(DEBUG, '=== Key exchange possibilities ===')
    for prefix, value in (('kex algos', kex_algo_list), ('server key', server_key_algo_list), ('client encrypt', client_encrypt_algo_list), ('server encrypt', server_encrypt_algo_list), ('client mac', client_mac_algo_list), ('server mac', server_mac_algo_list), ('client compress', client_compress_algo_list), ('server compress', server_compress_algo_list), ('client lang', client_lang_list), ('server lang', server_lang_list)):
        if value == ['']:
            value = ['<none>']
        value = ', '.join(value)
        self._log(DEBUG, '{}: {}'.format(prefix, value))
    self._log(DEBUG, 'kex follows: {}'.format(kex_follows))
    self._log(DEBUG, '=== Key exchange agreements ===')
    self._remote_ext_info = None
    self._remote_strict_kex = None
    to_pop = []
    for i, algo in enumerate(kex_algo_list):
        if algo.startswith('ext-info-'):
            self._remote_ext_info = algo
            to_pop.insert(0, i)
        elif algo.startswith('kex-strict-'):
            which = 'c' if self.server_mode else 's'
            expected = f'kex-strict-{which}-v00@openssh.com'
            self.agreed_on_strict_kex = algo == expected and self.advertise_strict_kex
            self._log(DEBUG, f'Strict kex mode: {self.agreed_on_strict_kex}')
            to_pop.insert(0, i)
    for i in to_pop:
        kex_algo_list.pop(i)
    if self.agreed_on_strict_kex and (not self.initial_kex_done) and (m.seqno != 0):
        raise MessageOrderError('In strict-kex mode, but KEXINIT was not the first packet!')
    if self.server_mode:
        agreed_kex = list(filter(self.preferred_kex.__contains__, kex_algo_list))
    else:
        agreed_kex = list(filter(kex_algo_list.__contains__, self.preferred_kex))
    if len(agreed_kex) == 0:
        raise IncompatiblePeer('Incompatible ssh peer (no acceptable kex algorithm)')
    self.kex_engine = self._kex_info[agreed_kex[0]](self)
    self._log(DEBUG, 'Kex: {}'.format(agreed_kex[0]))
    if self.server_mode:
        available_server_keys = list(filter(list(self.server_key_dict.keys()).__contains__, self.preferred_keys))
        agreed_keys = list(filter(available_server_keys.__contains__, server_key_algo_list))
    else:
        agreed_keys = list(filter(server_key_algo_list.__contains__, self.preferred_keys))
    if len(agreed_keys) == 0:
        raise IncompatiblePeer('Incompatible ssh peer (no acceptable host key)')
    self.host_key_type = agreed_keys[0]
    if self.server_mode and self.get_server_key() is None:
        raise IncompatiblePeer("Incompatible ssh peer (can't match requested host key type)")
    self._log_agreement('HostKey', agreed_keys[0], agreed_keys[0])
    if self.server_mode:
        agreed_local_ciphers = list(filter(self.preferred_ciphers.__contains__, server_encrypt_algo_list))
        agreed_remote_ciphers = list(filter(self.preferred_ciphers.__contains__, client_encrypt_algo_list))
    else:
        agreed_local_ciphers = list(filter(client_encrypt_algo_list.__contains__, self.preferred_ciphers))
        agreed_remote_ciphers = list(filter(server_encrypt_algo_list.__contains__, self.preferred_ciphers))
    if len(agreed_local_ciphers) == 0 or len(agreed_remote_ciphers) == 0:
        raise IncompatiblePeer('Incompatible ssh server (no acceptable ciphers)')
    self.local_cipher = agreed_local_ciphers[0]
    self.remote_cipher = agreed_remote_ciphers[0]
    self._log_agreement('Cipher', local=self.local_cipher, remote=self.remote_cipher)
    if self.server_mode:
        agreed_remote_macs = list(filter(self.preferred_macs.__contains__, client_mac_algo_list))
        agreed_local_macs = list(filter(self.preferred_macs.__contains__, server_mac_algo_list))
    else:
        agreed_local_macs = list(filter(client_mac_algo_list.__contains__, self.preferred_macs))
        agreed_remote_macs = list(filter(server_mac_algo_list.__contains__, self.preferred_macs))
    if len(agreed_local_macs) == 0 or len(agreed_remote_macs) == 0:
        raise IncompatiblePeer('Incompatible ssh server (no acceptable macs)')
    self.local_mac = agreed_local_macs[0]
    self.remote_mac = agreed_remote_macs[0]
    self._log_agreement('MAC', local=self.local_mac, remote=self.remote_mac)
    if self.server_mode:
        agreed_remote_compression = list(filter(self.preferred_compression.__contains__, client_compress_algo_list))
        agreed_local_compression = list(filter(self.preferred_compression.__contains__, server_compress_algo_list))
    else:
        agreed_local_compression = list(filter(client_compress_algo_list.__contains__, self.preferred_compression))
        agreed_remote_compression = list(filter(server_compress_algo_list.__contains__, self.preferred_compression))
    if len(agreed_local_compression) == 0 or len(agreed_remote_compression) == 0:
        msg = 'Incompatible ssh server (no acceptable compression)'
        msg += ' {!r} {!r} {!r}'
        raise IncompatiblePeer(msg.format(agreed_local_compression, agreed_remote_compression, self.preferred_compression))
    self.local_compression = agreed_local_compression[0]
    self.remote_compression = agreed_remote_compression[0]
    self._log_agreement('Compression', local=self.local_compression, remote=self.remote_compression)
    self._log(DEBUG, '=== End of kex handshake ===')
    self.remote_kex_init = cMSG_KEXINIT + m.get_so_far()