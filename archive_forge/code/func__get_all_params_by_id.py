from __future__ import (absolute_import, division, print_function)
import copy
import json
from ansible_collections.theforeman.foreman.plugins.module_utils._version import LooseVersion
from time import sleep
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name, Constructable
def _get_all_params_by_id(self, hid):
    url = '%s/api/v2/hosts/%s' % (self.foreman_url, hid)
    ret = self._get_json(url, [404])
    if not ret or not isinstance(ret, MutableMapping) or (not ret.get('all_parameters', False)):
        return {}
    return ret.get('all_parameters')