from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def _readinto_from_buf(self, b, index, length):
    cpy = min(length - index, len(self._buf))
    if cpy:
        b[index:index + cpy] = self._buf[:cpy]
        self._buf = self._buf[cpy:]
        index += cpy
    return index