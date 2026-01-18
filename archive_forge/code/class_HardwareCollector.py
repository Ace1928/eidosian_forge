from __future__ import (absolute_import, division, print_function)
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
class HardwareCollector(BaseFactCollector):
    name = 'hardware'
    _fact_ids = set(['processor', 'processor_cores', 'processor_count', 'mounts', 'devices'])
    _fact_class = Hardware

    def collect(self, module=None, collected_facts=None):
        collected_facts = collected_facts or {}
        if not module:
            return {}
        facts_obj = self._fact_class(module)
        facts_dict = facts_obj.populate(collected_facts=collected_facts)
        return facts_dict