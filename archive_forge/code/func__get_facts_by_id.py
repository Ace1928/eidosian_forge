from __future__ import (absolute_import, division, print_function)
import copy
import json
from ansible_collections.theforeman.foreman.plugins.module_utils._version import LooseVersion
from time import sleep
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name, Constructable
def _get_facts_by_id(self, hid):
    url = '%s/api/v2/hosts/%s/facts' % (self.foreman_url, hid)
    return self._get_json(url)