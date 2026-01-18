from __future__ import (absolute_import, division, print_function)
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
class VirtualCollector(BaseFactCollector):
    name = 'virtual'
    _fact_class = Virtual
    _fact_ids = set(['virtualization_type', 'virtualization_role', 'virtualization_tech_guest', 'virtualization_tech_host'])

    def collect(self, module=None, collected_facts=None):
        collected_facts = collected_facts or {}
        if not module:
            return {}
        facts_obj = self._fact_class(module)
        facts_dict = facts_obj.populate(collected_facts=collected_facts)
        return facts_dict