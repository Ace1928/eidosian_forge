from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
class DistributionFactCollector(BaseFactCollector):
    name = 'distribution'
    _fact_ids = set(['distribution_version', 'distribution_release', 'distribution_major_version', 'os_family'])

    def collect(self, module=None, collected_facts=None):
        collected_facts = collected_facts or {}
        facts_dict = {}
        if not module:
            return facts_dict
        distribution = Distribution(module=module)
        distro_facts = distribution.get_distribution_facts()
        return distro_facts