from __future__ import absolute_import, division, print_function
import os
import platform
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def add_dry_run_option(opts):
    if platform.system().lower() in ['openbsd', 'netbsd', 'freebsd']:
        opts.append('--check')
    else:
        opts.append('--dry-run')