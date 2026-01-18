from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def _canonify_root_arn(arn):
    if arn.startswith('arn:aws:iam::') and arn.endswith(':root'):
        arn = arn.split(':')[4]
    return arn