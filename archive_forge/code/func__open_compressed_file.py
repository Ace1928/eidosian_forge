from __future__ import absolute_import, division, print_function
import abc
import bz2
import glob
import gzip
import io
import os
import re
import shutil
import tarfile
import zipfile
from fnmatch import fnmatch
from sys import version_info
from traceback import format_exc
from zlib import crc32
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils import six
def _open_compressed_file(self, path, mode):
    f = None
    if self.format == 'gz':
        f = gzip.open(path, mode)
    elif self.format == 'bz2':
        f = bz2.BZ2File(path, mode)
    elif self.format == 'xz':
        f = lzma.LZMAFile(path, mode)
    else:
        self.module.fail_json(msg='%s is not a valid format' % self.format)
    return f