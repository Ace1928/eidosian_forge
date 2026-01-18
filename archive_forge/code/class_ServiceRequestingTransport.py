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
class ServiceRequestingTransport(Transport):
    """
    Transport, but also handling service requests, like it oughtta!

    .. versionadded:: 3.2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_userauth_accepted = False
        self._handler_table[MSG_SERVICE_ACCEPT] = self._parse_service_accept

    def _parse_service_accept(self, m):
        service = m.get_text()
        if service != 'ssh-userauth':
            self._log(DEBUG, 'Service request "{}" accepted (?)'.format(service))
            return
        self._service_userauth_accepted = True
        self._log(DEBUG, 'MSG_SERVICE_ACCEPT received; auth may begin')

    def ensure_session(self):
        if not self.active or not self.initial_kex_done:
            raise SSHException('No existing session')
        if self._service_userauth_accepted:
            return
        m = Message()
        m.add_byte(cMSG_SERVICE_REQUEST)
        m.add_string('ssh-userauth')
        self._log(DEBUG, 'Sending MSG_SERVICE_REQUEST: ssh-userauth')
        self._send_message(m)
        while not self._service_userauth_accepted:
            time.sleep(0.1)
        self.auth_handler = self.get_auth_handler()

    def get_auth_handler(self):
        return AuthOnlyHandler(self)

    def auth_none(self, username):
        self.ensure_session()
        return self.auth_handler.auth_none(username)

    def auth_password(self, username, password, fallback=True):
        self.ensure_session()
        try:
            return self.auth_handler.auth_password(username, password)
        except BadAuthenticationType as e:
            if not fallback or 'keyboard-interactive' not in e.allowed_types:
                raise
            try:

                def handler(title, instructions, fields):
                    if len(fields) > 1:
                        raise SSHException('Fallback authentication failed.')
                    if len(fields) == 0:
                        return []
                    return [password]
                return self.auth_interactive(username, handler)
            except SSHException:
                raise e

    def auth_publickey(self, username, key):
        self.ensure_session()
        return self.auth_handler.auth_publickey(username, key)

    def auth_interactive(self, username, handler, submethods=''):
        self.ensure_session()
        return self.auth_handler.auth_interactive(username, handler, submethods)

    def auth_interactive_dumb(self, username, handler=None, submethods=''):
        self.ensure_session()
        if not handler:

            def handler(title, instructions, prompt_list):
                answers = []
                if title:
                    print(title.strip())
                if instructions:
                    print(instructions.strip())
                for prompt, show_input in prompt_list:
                    print(prompt.strip(), end=' ')
                    answers.append(input())
                return answers
        return self.auth_interactive(username, handler, submethods)

    def auth_gssapi_with_mic(self, username, gss_host, gss_deleg_creds):
        self.ensure_session()
        self.auth_handler = self.get_auth_handler()
        return self.auth_handler.auth_gssapi_with_mic(username, gss_host, gss_deleg_creds)

    def auth_gssapi_keyex(self, username):
        self.ensure_session()
        self.auth_handler = self.get_auth_handler()
        return self.auth_handler.auth_gssapi_keyex(username)