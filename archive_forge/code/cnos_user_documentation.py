from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import run_commands, load_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import get_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types, iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import get_user_roles
 main entry point for module execution
    