from __future__ import absolute_import, division, print_function
import os
import json
import tempfile
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.six import integer_types
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _state_args(state_file):
    if not state_file:
        return []
    if not os.path.exists(state_file):
        module.warn('Could not find state_file "{0}", the process will not destroy any resources, please check your state file path.'.format(state_file))
    return ['-state', state_file]