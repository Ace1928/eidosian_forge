from __future__ import (absolute_import, division, print_function)
import abc
import re
from os.path import basename
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _matches_regex(self, proc, regex):
    attributes = self._get_proc_attributes(proc, *self.PATTERN_ATTRS)
    matches_name = regex.search(to_native(attributes['name']))
    matches_exe = attributes['exe'] and regex.search(basename(to_native(attributes['exe'])))
    matches_cmd = attributes['cmdline'] and regex.search(to_native(' '.join(attributes['cmdline'])))
    return any([matches_name, matches_exe, matches_cmd])