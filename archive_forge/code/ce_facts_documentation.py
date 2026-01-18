from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import run_commands
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
 Populate method