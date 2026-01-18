from __future__ import absolute_import, division, print_function
import time
from ansible_collections.community.network.plugins.module_utils.network.aruba.aruba import run_commands
from ansible_collections.community.network.plugins.module_utils.network.aruba.aruba import aruba_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Conditional
from ansible.module_utils.six import string_types
main entry point for module execution
    