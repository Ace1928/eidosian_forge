from __future__ import (absolute_import, division, print_function)
import os
import ansible.module_utils.compat.typing as t
from ansible.module_utils.six import iteritems
from ansible.module_utils.facts.collector import BaseFactCollector
class EnvFactCollector(BaseFactCollector):
    name = 'env'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        env_facts = {}
        env_facts['env'] = {}
        for k, v in iteritems(os.environ):
            env_facts['env'][k] = v
        return env_facts