from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _create_regex_group_complement(s):
    lines = (line.strip() for line in s.split('\n') if line.strip())
    chars = filter(None, (line.split('#')[0].strip() for line in lines))
    group = '[^' + ''.join(chars) + ']'
    return re.compile(group)