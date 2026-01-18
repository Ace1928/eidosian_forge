from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def _do_builds_differ(current, desired):
    if current is None:
        return True
    if len(current) != len(desired):
        return True
    return _build_set(current) != _build_set(desired)