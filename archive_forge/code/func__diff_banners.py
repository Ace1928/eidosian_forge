from __future__ import absolute_import, division, print_function
import json
import re
import time
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import (
def _diff_banners(self, want, have):
    candidate = {}
    for key, value in iteritems(want):
        if value != have.get(key):
            candidate[key] = value
    return candidate