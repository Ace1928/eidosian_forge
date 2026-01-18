from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def _fix_template_vars(message):
    if message:
        return message.replace('[[', '{{').replace(']]', '}}')
    return message