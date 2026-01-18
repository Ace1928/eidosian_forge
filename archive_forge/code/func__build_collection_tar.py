from __future__ import (absolute_import, division, print_function)
import errno
import fnmatch
import functools
import json
import os
import pathlib
import queue
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import typing as t
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, fields as dc_fields
from hashlib import sha256
from io import BytesIO
from importlib.metadata import distribution
from itertools import chain
import ansible.constants as C
from ansible.compat.importlib_resources import files
from ansible.errors import AnsibleError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.galaxy_api_proxy import MultiGalaxyAPIProxy
from ansible.galaxy.collection.gpg import (
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import meets_requirements
from ansible.plugins.loader import get_all_plugin_loaders
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_dump
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash, secure_hash_s
from ansible.utils.sentinel import Sentinel
def _build_collection_tar(b_collection_path, b_tar_path, collection_manifest, file_manifest):
    """Build a tar.gz collection artifact from the manifest data."""
    files_manifest_json = to_bytes(json.dumps(file_manifest, indent=True), errors='surrogate_or_strict')
    collection_manifest['file_manifest_file']['chksum_sha256'] = secure_hash_s(files_manifest_json, hash_func=sha256)
    collection_manifest_json = to_bytes(json.dumps(collection_manifest, indent=True), errors='surrogate_or_strict')
    with _tempdir() as b_temp_path:
        b_tar_filepath = os.path.join(b_temp_path, os.path.basename(b_tar_path))
        with tarfile.open(b_tar_filepath, mode='w:gz') as tar_file:
            for name, b in [(MANIFEST_FILENAME, collection_manifest_json), ('FILES.json', files_manifest_json)]:
                b_io = BytesIO(b)
                tar_info = tarfile.TarInfo(name)
                tar_info.size = len(b)
                tar_info.mtime = int(time.time())
                tar_info.mode = 420
                tar_file.addfile(tarinfo=tar_info, fileobj=b_io)
            for file_info in file_manifest['files']:
                if file_info['name'] == '.':
                    continue
                filename = to_native(file_info['name'], errors='surrogate_or_strict')
                b_src_path = os.path.join(b_collection_path, to_bytes(filename, errors='surrogate_or_strict'))

                def reset_stat(tarinfo):
                    if tarinfo.type != tarfile.SYMTYPE:
                        existing_is_exec = tarinfo.mode & stat.S_IXUSR
                        tarinfo.mode = 493 if existing_is_exec or tarinfo.isdir() else 420
                    tarinfo.uid = tarinfo.gid = 0
                    tarinfo.uname = tarinfo.gname = ''
                    return tarinfo
                if os.path.islink(b_src_path):
                    b_link_target = os.path.realpath(b_src_path)
                    if not os.path.exists(b_link_target):
                        raise AnsibleError(f"Failed to find the target path '{to_native(b_link_target)}' for the symlink '{to_native(b_src_path)}'.")
                    if _is_child_path(b_link_target, b_collection_path):
                        b_rel_path = os.path.relpath(b_link_target, start=os.path.dirname(b_src_path))
                        tar_info = tarfile.TarInfo(filename)
                        tar_info.type = tarfile.SYMTYPE
                        tar_info.linkname = to_native(b_rel_path, errors='surrogate_or_strict')
                        tar_info = reset_stat(tar_info)
                        tar_file.addfile(tarinfo=tar_info)
                        continue
                tar_file.add(to_native(os.path.realpath(b_src_path)), arcname=filename, recursive=False, filter=reset_stat)
        shutil.copy(to_native(b_tar_filepath), to_native(b_tar_path))
        collection_name = '%s.%s' % (collection_manifest['collection_info']['namespace'], collection_manifest['collection_info']['name'])
        tar_path = to_text(b_tar_path)
        display.display(u'Created collection for %s at %s' % (collection_name, tar_path))
        return tar_path