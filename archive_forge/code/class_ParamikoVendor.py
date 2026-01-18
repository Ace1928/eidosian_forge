import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
class ParamikoVendor(SSHVendor):
    """Vendor that uses paramiko."""

    def _hexify(self, s):
        return hexlify(s).upper()

    def _connect(self, username, password, host, port):
        global SYSTEM_HOSTKEYS, BRZ_HOSTKEYS
        load_host_keys()
        try:
            t = paramiko.Transport((host, port or 22))
            t.set_log_channel('bzr.paramiko')
            t.start_client()
        except (paramiko.SSHException, OSError) as e:
            self._raise_connection_error(host, port=port, orig_error=e)
        server_key = t.get_remote_server_key()
        server_key_hex = self._hexify(server_key.get_fingerprint())
        keytype = server_key.get_name()
        if host in SYSTEM_HOSTKEYS and keytype in SYSTEM_HOSTKEYS[host]:
            our_server_key = SYSTEM_HOSTKEYS[host][keytype]
            our_server_key_hex = self._hexify(our_server_key.get_fingerprint())
        elif host in BRZ_HOSTKEYS and keytype in BRZ_HOSTKEYS[host]:
            our_server_key = BRZ_HOSTKEYS[host][keytype]
            our_server_key_hex = self._hexify(our_server_key.get_fingerprint())
        else:
            trace.warning('Adding %s host key for %s: %s' % (keytype, host, server_key_hex))
            add = getattr(BRZ_HOSTKEYS, 'add', None)
            if add is not None:
                BRZ_HOSTKEYS.add(host, keytype, server_key)
            else:
                BRZ_HOSTKEYS.setdefault(host, {})[keytype] = server_key
            our_server_key = server_key
            our_server_key_hex = self._hexify(our_server_key.get_fingerprint())
            save_host_keys()
        if server_key != our_server_key:
            filename1 = os.path.expanduser('~/.ssh/known_hosts')
            filename2 = _ssh_host_keys_config_dir()
            raise errors.TransportError('Host keys for %s do not match!  %s != %s' % (host, our_server_key_hex, server_key_hex), ['Try editing {} or {}'.format(filename1, filename2)])
        _paramiko_auth(username, password, host, port, t)
        return t

    def connect_sftp(self, username, password, host, port):
        t = self._connect(username, password, host, port)
        try:
            return t.open_sftp_client()
        except paramiko.SSHException as e:
            self._raise_connection_error(host, port=port, orig_error=e, msg='Unable to start sftp client')

    def connect_ssh(self, username, password, host, port, command):
        t = self._connect(username, password, host, port)
        try:
            channel = t.open_session()
            cmdline = ' '.join(command)
            channel.exec_command(cmdline)
            return _ParamikoSSHConnection(channel)
        except paramiko.SSHException as e:
            self._raise_connection_error(host, port=port, orig_error=e, msg='Unable to invoke remote bzr')