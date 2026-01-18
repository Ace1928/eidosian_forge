from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _set_value_detach_interactive(module, data, api_version, options, values):
    interactive = values.get('interactive')
    detach = values.get('detach')
    data['AttachStdout'] = False
    data['AttachStderr'] = False
    data['AttachStdin'] = False
    data['StdinOnce'] = False
    data['OpenStdin'] = interactive
    if not detach:
        data['AttachStdout'] = True
        data['AttachStderr'] = True
        if interactive:
            data['AttachStdin'] = True
            data['StdinOnce'] = True