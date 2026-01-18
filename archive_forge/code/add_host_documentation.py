from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping
from ansible.errors import AnsibleActionFail
from ansible.module_utils.six import string_types
from ansible.plugins.action import ActionBase
from ansible.parsing.utils.addresses import parse_address
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
 Create inventory hosts and groups in the memory inventory