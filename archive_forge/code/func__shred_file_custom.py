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
def _shred_file_custom(self, tmp_path):
    """"Destroy a file, when shred (core-utils) is not available

        Unix `shred' destroys files "so that they can be recovered only with great difficulty with
        specialised hardware, if at all". It is based on the method from the paper
        "Secure Deletion of Data from Magnetic and Solid-State Memory",
        Proceedings of the Sixth USENIX Security Symposium (San Jose, California, July 22-25, 1996).

        We do not go to that length to re-implement shred in Python; instead, overwriting with a block
        of random data should suffice.

        See https://github.com/ansible/ansible/pull/13700 .
        """
    file_len = os.path.getsize(tmp_path)
    if file_len > 0:
        max_chunk_len = min(1024 * 1024 * 2, file_len)
        passes = 3
        with open(tmp_path, 'wb') as fh:
            for dummy in range(passes):
                fh.seek(0, 0)
                chunk_len = random.randint(max_chunk_len // 2, max_chunk_len)
                data = os.urandom(chunk_len)
                for dummy in range(0, file_len // chunk_len):
                    fh.write(data)
                fh.write(data[:file_len % chunk_len])
                if fh.tell() != file_len:
                    raise AnsibleAssertionError()
                os.fsync(fh)