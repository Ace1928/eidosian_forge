from __future__ import (absolute_import, division, print_function)
import fnmatch
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts import collector
from ansible.module_utils.common.collections import is_string
class AnsibleFactCollector(collector.BaseFactCollector):
    """A FactCollector that returns results under 'ansible_facts' top level key.

       If a namespace if provided, facts will be collected under that namespace.
       For ex, a ansible.module_utils.facts.namespace.PrefixFactNamespace(prefix='ansible_')

       Has a 'from_gather_subset() constructor that populates collectors based on a
       gather_subset specifier."""

    def __init__(self, collectors=None, namespace=None, filter_spec=None):
        super(AnsibleFactCollector, self).__init__(collectors=collectors, namespace=namespace)
        self.filter_spec = filter_spec

    def _filter(self, facts_dict, filter_spec):
        if not filter_spec or filter_spec == '*':
            return facts_dict
        if is_string(filter_spec):
            filter_spec = [filter_spec]
        found = []
        for f in filter_spec:
            for x, y in facts_dict.items():
                if not f or fnmatch.fnmatch(x, f):
                    found.append((x, y))
                elif not f.startswith(('ansible_', 'facter', 'ohai')):
                    g = 'ansible_%s' % f
                    if fnmatch.fnmatch(x, g):
                        found.append((x, y))
        return found

    def collect(self, module=None, collected_facts=None):
        collected_facts = collected_facts or {}
        facts_dict = {}
        for collector_obj in self.collectors:
            info_dict = {}
            try:
                info_dict = collector_obj.collect_with_namespace(module=module, collected_facts=collected_facts)
            except Exception as e:
                sys.stderr.write(repr(e))
                sys.stderr.write('\n')
            collected_facts.update(info_dict.copy())
            facts_dict.update(self._filter(info_dict, self.filter_spec))
        return facts_dict