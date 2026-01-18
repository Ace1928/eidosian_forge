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
def _regular_content_tar_generator(content, out_file, user_id, group_id, mode, user_name=None):
    tarinfo = tarfile.TarInfo()
    tarinfo.name = os.path.splitdrive(to_text(out_file))[1].replace(os.sep, '/').lstrip('/')
    tarinfo.mode = mode
    tarinfo.uid = user_id
    tarinfo.gid = group_id
    tarinfo.size = len(content)
    try:
        tarinfo.mtime = int(datetime.datetime.now().timestamp())
    except AttributeError:
        tarinfo.mtime = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds())
    tarinfo.type = tarfile.REGTYPE
    tarinfo.linkname = ''
    if user_name:
        tarinfo.uname = user_name
    tarinfo_buf = tarinfo.tobuf()
    total_size = len(tarinfo_buf)
    yield tarinfo_buf
    total_size += len(content)
    yield content
    remainder = tarinfo.size % tarfile.BLOCKSIZE
    if remainder:
        yield (tarfile.NUL * (tarfile.BLOCKSIZE - remainder))
        total_size += tarfile.BLOCKSIZE - remainder
    yield (tarfile.NUL * (2 * tarfile.BLOCKSIZE))
    total_size += 2 * tarfile.BLOCKSIZE
    remainder = total_size % tarfile.RECORDSIZE
    if remainder > 0:
        yield (tarfile.NUL * (tarfile.RECORDSIZE - remainder))