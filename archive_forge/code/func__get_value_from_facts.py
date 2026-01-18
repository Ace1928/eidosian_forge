from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _get_value_from_facts(self, variable_name, distribution, default_value):
    """Get dist+version specific args first, then distribution, then family, lastly use default"""
    attr = getattr(self, variable_name)
    value = attr.get(distribution['name'] + distribution['version'], attr.get(distribution['name'], attr.get(distribution['family'], getattr(self, default_value))))
    return value