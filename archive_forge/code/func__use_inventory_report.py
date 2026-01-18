from __future__ import (absolute_import, division, print_function)
import copy
import json
from ansible_collections.theforeman.foreman.plugins.module_utils._version import LooseVersion
from time import sleep
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name, Constructable
def _use_inventory_report(self):
    use_inventory_report = self.get_option('use_reports_api')
    try:
        use_inventory_report = self.get_option('foreman').get('use_reports_api')
    except Exception:
        pass
    if not use_inventory_report:
        return False
    status_url = '%s/api/v2/status' % self.foreman_url
    result = self._get_json(status_url)
    foreman_version = LooseVersion(result.get('version')) >= LooseVersion(self.MINIMUM_FOREMAN_VERSION_FOR_REPORTS_API)
    return foreman_version