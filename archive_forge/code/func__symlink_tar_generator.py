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
def _symlink_tar_generator(b_in_path, file_stat, out_file, user_id, group_id, mode=None, user_name=None):
    yield _symlink_tar_creator(b_in_path, file_stat, out_file, user_id, group_id, mode, user_name)