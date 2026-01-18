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
def _get_value_detach_interactive(module, container, api_version, options, image, host_info):
    attach_stdin = container['Config'].get('OpenStdin')
    attach_stderr = container['Config'].get('AttachStderr')
    attach_stdout = container['Config'].get('AttachStdout')
    return {'interactive': bool(attach_stdin), 'detach': not (attach_stderr and attach_stdout)}