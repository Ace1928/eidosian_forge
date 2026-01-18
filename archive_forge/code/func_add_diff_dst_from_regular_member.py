from __future__ import absolute_import, division, print_function
import base64
import io
import os
import stat
import traceback
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.module_utils._scramble import generate_insecure_key, scramble
def add_diff_dst_from_regular_member(diff, max_file_size_for_diff, container_path, tar, member):
    if diff is None:
        return
    if member.size > max_file_size_for_diff > 0:
        diff['dst_larger'] = max_file_size_for_diff
        return
    tar_f = tar.extractfile(member)
    content = tar_f.read()
    if is_binary(content):
        diff['dst_binary'] = 1
    else:
        diff['before_header'] = container_path
        diff['before'] = to_text(content)