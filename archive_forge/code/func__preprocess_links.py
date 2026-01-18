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
def _preprocess_links(module, client, api_version, value):
    if value is None:
        return None
    result = []
    for link in value:
        parsed_link = link.split(':', 1)
        if len(parsed_link) == 2:
            link, alias = parsed_link
        else:
            link, alias = (parsed_link[0], parsed_link[0])
        result.append('/%s:/%s/%s' % (link, module.params['name'], alias))
    return result