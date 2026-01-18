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
def _build_files_manifest_walk(b_collection_path, namespace, name, ignore_patterns):
    b_ignore_patterns = [b'MANIFEST.json', b'FILES.json', b'galaxy.yml', b'galaxy.yaml', b'.git', b'*.pyc', b'*.retry', b'tests/output', to_bytes('{0}-{1}-*.tar.gz'.format(namespace, name))]
    b_ignore_patterns += [to_bytes(p) for p in ignore_patterns]
    b_ignore_dirs = frozenset([b'CVS', b'.bzr', b'.hg', b'.git', b'.svn', b'__pycache__', b'.tox'])
    manifest = _make_manifest()

    def _walk(b_path, b_top_level_dir):
        for b_item in os.listdir(b_path):
            b_abs_path = os.path.join(b_path, b_item)
            b_rel_base_dir = b'' if b_path == b_top_level_dir else b_path[len(b_top_level_dir) + 1:]
            b_rel_path = os.path.join(b_rel_base_dir, b_item)
            rel_path = to_text(b_rel_path, errors='surrogate_or_strict')
            if os.path.isdir(b_abs_path):
                if any((b_item == b_path for b_path in b_ignore_dirs)) or any((fnmatch.fnmatch(b_rel_path, b_pattern) for b_pattern in b_ignore_patterns)):
                    display.vvv("Skipping '%s' for collection build" % to_text(b_abs_path))
                    continue
                if os.path.islink(b_abs_path):
                    b_link_target = os.path.realpath(b_abs_path)
                    if not _is_child_path(b_link_target, b_top_level_dir):
                        display.warning("Skipping '%s' as it is a symbolic link to a directory outside the collection" % to_text(b_abs_path))
                        continue
                manifest['files'].append(_make_entry(rel_path, 'dir'))
                if not os.path.islink(b_abs_path):
                    _walk(b_abs_path, b_top_level_dir)
            else:
                if any((fnmatch.fnmatch(b_rel_path, b_pattern) for b_pattern in b_ignore_patterns)):
                    display.vvv("Skipping '%s' for collection build" % to_text(b_abs_path))
                    continue
                manifest['files'].append(_make_entry(rel_path, 'file', chksum_type='sha256', chksum=secure_hash(b_abs_path, hash_func=sha256)))
    _walk(b_collection_path, b_collection_path)
    return manifest