from __future__ import absolute_import, division, print_function
import hashlib
import os
import posixpath
import shutil
import io
import tempfile
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.ansible_release import __version__ as ansible_version
from re import match
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def _local_checksum(self, checksum_alg, file):
    if checksum_alg.lower() == 'md5':
        hash = hashlib.md5()
    elif checksum_alg.lower() == 'sha1':
        hash = hashlib.sha1()
    else:
        raise ValueError('Unknown checksum_alg %s' % checksum_alg)
    with io.open(file, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash.update(chunk)
    return hash.hexdigest()