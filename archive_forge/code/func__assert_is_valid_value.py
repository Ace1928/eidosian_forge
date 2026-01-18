from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _assert_is_valid_value(module, item, value, prefix=''):
    if item in ['nice', 'priority']:
        try:
            valid = -20 <= int(value) <= 19
        except ValueError:
            valid = False
        if not valid:
            module.fail_json(msg='%s Value of %r for item %r is invalid. Value must be a number in the range -20 to 19 inclusive. Refer to the limits.conf(5) manual pages for more details.' % (prefix, value, item))
    elif not (value in ['unlimited', 'infinity', '-1'] or value.isdigit()):
        module.fail_json(msg="%s Value of %r for item %r is invalid. Value must either be 'unlimited', 'infinity' or -1, all of which indicate no limit, or a limit of 0 or larger. Refer to the limits.conf(5) manual pages for more details." % (prefix, value, item))