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
def _regular_file_tar_generator(b_in_path, file_stat, out_file, user_id, group_id, mode=None, user_name=None):
    if not stat.S_ISREG(file_stat.st_mode):
        raise DockerUnexpectedError('stat information is not for a regular file')
    tarinfo = tarfile.TarInfo()
    tarinfo.name = os.path.splitdrive(to_text(out_file))[1].replace(os.sep, '/').lstrip('/')
    tarinfo.mode = file_stat.st_mode & 448 if mode is None else mode
    tarinfo.uid = user_id
    tarinfo.gid = group_id
    tarinfo.size = file_stat.st_size
    tarinfo.mtime = file_stat.st_mtime
    tarinfo.type = tarfile.REGTYPE
    tarinfo.linkname = ''
    if user_name:
        tarinfo.uname = user_name
    tarinfo_buf = tarinfo.tobuf()
    total_size = len(tarinfo_buf)
    yield tarinfo_buf
    size = tarinfo.size
    total_size += size
    with open(b_in_path, 'rb') as f:
        while size > 0:
            to_read = min(size, 65536)
            buf = f.read(to_read)
            if not buf:
                break
            size -= len(buf)
            yield buf
    if size:
        yield (tarfile.NUL * size)
    remainder = tarinfo.size % tarfile.BLOCKSIZE
    if remainder:
        yield (tarfile.NUL * (tarfile.BLOCKSIZE - remainder))
        total_size += tarfile.BLOCKSIZE - remainder
    yield (tarfile.NUL * (2 * tarfile.BLOCKSIZE))
    total_size += 2 * tarfile.BLOCKSIZE
    remainder = total_size % tarfile.RECORDSIZE
    if remainder > 0:
        yield (tarfile.NUL * (tarfile.RECORDSIZE - remainder))