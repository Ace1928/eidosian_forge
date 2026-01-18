from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
class FileVaultSecret(VaultSecret):

    def __init__(self, filename=None, encoding=None, loader=None):
        super(FileVaultSecret, self).__init__()
        self.filename = filename
        self.loader = loader
        self.encoding = encoding or 'utf8'
        self._bytes = None
        self._text = None

    @property
    def bytes(self):
        if self._bytes:
            return self._bytes
        if self._text:
            return self._text.encode(self.encoding)
        return None

    def load(self):
        self._bytes = self._read_file(self.filename)

    def _read_file(self, filename):
        """
        Read a vault password from a file or if executable, execute the script and
        retrieve password from STDOUT
        """
        try:
            with open(filename, 'rb') as f:
                vault_pass = f.read().strip()
        except (OSError, IOError) as e:
            raise AnsibleError('Could not read vault password file %s: %s' % (filename, e))
        b_vault_data, dummy = self.loader._decrypt_if_vault_data(vault_pass, filename)
        vault_pass = b_vault_data.strip(b'\r\n')
        verify_secret_is_not_empty(vault_pass, msg='Invalid vault password was provided from file (%s)' % filename)
        return vault_pass

    def __repr__(self):
        if self.filename:
            return "%s(filename='%s')" % (self.__class__.__name__, self.filename)
        return '%s()' % self.__class__.__name__