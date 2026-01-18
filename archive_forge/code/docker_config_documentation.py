from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.common.text.converters import to_native, to_bytes
 Handles state == 'absent', removing the config 