from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems

    Wrapper for ``AnsibleModule.run_command()``.

    It aims to provide a reusable runner with consistent argument formatting
    and sensible defaults.
    