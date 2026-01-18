from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
import subprocess
import typing as t
from contextlib import contextmanager
from hashlib import sha256
from urllib.error import URLError
from urllib.parse import urldefrag
from shutil import rmtree
from tempfile import mkdtemp
from ansible.errors import AnsibleError
from ansible.galaxy import get_collections_galaxy_meta_info
from ansible.galaxy.api import should_retry_error
from ansible.galaxy.dependency_resolution.dataclasses import _GALAXY_YAML
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
import yaml
def _get_meta_from_installed_dir(b_path):
    manifest = _get_json_from_installed_dir(b_path, MANIFEST_FILENAME)
    collection_info = manifest['collection_info']
    version = collection_info.get('version')
    if not version:
        raise AnsibleError(u'Collection metadata file `{manifest_filename!s}` at `{meta_file!s}` is expected to have a valid SemVer version value but got {version!s}'.format(manifest_filename=MANIFEST_FILENAME, meta_file=to_text(b_path), version=to_text(repr(version))))
    return collection_info