from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def add_afi(entry, afi):
    """adds afi needed for
            setval processing"""
    if entry:
        entry['afi'] = afi
    return entry