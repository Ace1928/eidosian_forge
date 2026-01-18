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
def _edit_file_helper(self, filename, secret, existing_data=None, force_save=False, vault_id=None):
    root, ext = os.path.splitext(os.path.realpath(filename))
    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=C.DEFAULT_LOCAL_TMP)
    cmd = self._editor_shell_command(tmp_path)
    try:
        if existing_data:
            self.write_data(existing_data, fd, shred=False)
    except Exception:
        self._shred_file(tmp_path)
        raise
    finally:
        os.close(fd)
    try:
        subprocess.call(cmd)
    except Exception as e:
        self._shred_file(tmp_path)
        raise AnsibleError('Unable to execute the command "%s": %s' % (' '.join(cmd), to_native(e)))
    b_tmpdata = self.read_data(tmp_path)
    if force_save or existing_data != b_tmpdata:
        b_ciphertext = self.vault.encrypt(b_tmpdata, secret, vault_id=vault_id)
        self.write_data(b_ciphertext, tmp_path)
        self.shuffle_files(tmp_path, filename)
        display.vvvvv(u'Saved edited file "%s" encrypted using %s and  vault id "%s"' % (to_text(filename), to_text(secret), to_text(vault_id)))
    self._shred_file(tmp_path)