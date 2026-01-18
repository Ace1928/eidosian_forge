from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
class BaseFactCollector:
    _fact_ids = set()
    _platform = 'Generic'
    name = None
    required_facts = set()

    def __init__(self, collectors=None, namespace=None):
        """Base class for things that collect facts.

        'collectors' is an optional list of other FactCollectors for composing."""
        self.collectors = collectors or []
        self.namespace = namespace
        self.fact_ids = set([self.name])
        self.fact_ids.update(self._fact_ids)

    @classmethod
    def platform_match(cls, platform_info):
        if platform_info.get('system', None) == cls._platform:
            return cls
        return None

    def _transform_name(self, key_name):
        if self.namespace:
            return self.namespace.transform(key_name)
        return key_name

    def _transform_dict_keys(self, fact_dict):
        """update a dicts keys to use new names as transformed by self._transform_name"""
        for old_key in list(fact_dict.keys()):
            new_key = self._transform_name(old_key)
            fact_dict[new_key] = fact_dict.pop(old_key)
        return fact_dict

    def collect_with_namespace(self, module=None, collected_facts=None):
        facts_dict = self.collect(module=module, collected_facts=collected_facts)
        if self.namespace:
            facts_dict = self._transform_dict_keys(facts_dict)
        return facts_dict

    def collect(self, module=None, collected_facts=None):
        """do the fact collection

        'collected_facts' is a object (a dict, likely) that holds all previously
          facts. This is intended to be used if a FactCollector needs to reference
          another fact (for ex, the system arch) and should not be modified (usually).

          Returns a dict of facts.

          """
        facts_dict = {}
        return facts_dict