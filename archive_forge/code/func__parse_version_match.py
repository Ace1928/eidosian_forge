from __future__ import (absolute_import, division, print_function)
import base64
import errno
import json
import os
import pkgutil
import random
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.compat.importlib import import_module
from ansible.plugins.loader import ps_module_utils_loader
from ansible.utils.collection_loader import resource_from_fqcr
def _parse_version_match(self, match, attribute):
    new_version = to_text(match.group(1)).rstrip()
    if match.group(2) is None:
        new_version = '%s.0' % new_version
    existing_version = getattr(self, attribute, None)
    if existing_version is None:
        setattr(self, attribute, new_version)
    elif LooseVersion(new_version) > LooseVersion(existing_version):
        setattr(self, attribute, new_version)