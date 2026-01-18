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
def _really_parse_kex_init(self, m, ignore_first_byte=False):
    parsed = {}
    if ignore_first_byte:
        m.get_byte()
    m.get_bytes(16)
    parsed['kex_algo_list'] = m.get_list()
    parsed['server_key_algo_list'] = m.get_list()
    parsed['client_encrypt_algo_list'] = m.get_list()
    parsed['server_encrypt_algo_list'] = m.get_list()
    parsed['client_mac_algo_list'] = m.get_list()
    parsed['server_mac_algo_list'] = m.get_list()
    parsed['client_compress_algo_list'] = m.get_list()
    parsed['server_compress_algo_list'] = m.get_list()
    parsed['client_lang_list'] = m.get_list()
    parsed['server_lang_list'] = m.get_list()
    parsed['kex_follows'] = m.get_boolean()
    m.get_int()
    return parsed